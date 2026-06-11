from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from autogen_core.models import (
    AssistantMessage,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)

from .compiler import LocalPromptCompiler
from .ir import Movability, PromptBlock, ShareScope, stable_hash
from .planner import HierarchicalPrefixPlanner, rewrite_messages
from .semantic_candidates import (
    SemanticCandidate,
    extract_semantic_candidates,
    semantic_candidate_summary_to_dict,
    summarize_semantic_candidates,
)
from .telemetry import dataclass_to_dict, serialize_prefix_tree, summarize_telemetry
from .validator import CacheUtilityValidator


@dataclass(frozen=True)
class DatasetEvaluationResult:
    input_path: str
    telemetry_path: str | None
    summary_path: str | None
    summary: dict[str, Any]


def evaluate_dataset(
    *,
    input_path: str | Path,
    telemetry_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    session_id: str = "dataset-eval",
    max_records: int | None = None,
    enable_natural_language_segmentation: bool = False,
    enable_groupchat_history_reordering: bool = False,
) -> DatasetEvaluationResult:
    source_path = Path(input_path)
    source_records = tuple(_load_jsonl(source_path, max_records=max_records))
    compiler = LocalPromptCompiler(
        enable_natural_language_segmentation=enable_natural_language_segmentation,
        enable_groupchat_history_reordering=enable_groupchat_history_reordering,
    )
    planner = HierarchicalPrefixPlanner(enable_groupchat_history_reordering=enable_groupchat_history_reordering)
    validator = CacheUtilityValidator()

    telemetry_records: list[dict[str, Any]] = []
    skipped_reason_counts: Counter[str] = Counter()
    semantic_type_counts: Counter[str] = Counter()
    movability_counts: Counter[str] = Counter()
    share_scope_counts: Counter[str] = Counter()
    moved_semantic_type_counts: Counter[str] = Counter()
    repeated_system_blocks: dict[str, dict[str, Any]] = {}
    natural_language_hint_counts: Counter[str] = Counter()
    semantic_candidates_by_parent: dict[str, tuple[SemanticCandidate, ...]] = {}

    for record_index, record in enumerate(source_records, start=1):
        body = _extract_openai_request(record)
        if body is None:
            skipped_reason_counts["missing_messages"] += 1
            continue

        try:
            messages = openai_messages_to_autogen(body.get("messages"))
        except ValueError:
            skipped_reason_counts["invalid_messages"] += 1
            continue

        effective_session_id = str(record.get("session_id") or body.get("session_id") or session_id)
        tools = tuple(body.get("tools") or ())
        tool_choice = body.get("tool_choice", "auto")
        json_output = _json_output_arg(body)
        extra_create_args = _extra_create_args(body)

        compile_result = compiler.compile(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            session_id=effective_session_id,
        )
        plan = planner.plan(compile_result, session_id=effective_session_id)
        rewritten_messages = rewrite_messages(compile_result, plan)
        report = validator.validate(
            original_messages=messages,
            rewritten_messages=rewritten_messages,
            compile_result=compile_result,
            plan=plan,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
        )

        blocks_by_id = {block.block_id: block for block in compile_result.blocks}
        for block in compile_result.blocks:
            semantic_type_counts[block.semantic_type.value] += 1
            movability_counts[block.movability.value] += 1
            share_scope_counts[block.share_scope.value] += 1
            _observe_rule_gap_candidate(
                repeated_system_blocks,
                natural_language_hint_counts,
                semantic_candidates_by_parent,
                block=block,
                record_index=record_index,
            )
        for block_id in plan.moved_blocks:
            block = blocks_by_id.get(block_id)
            if block is not None:
                moved_semantic_type_counts[block.semantic_type.value] += 1

        telemetry_records.append(
            _build_eval_record(
                record=record,
                record_index=record_index,
                effective_session_id=effective_session_id,
                messages=messages,
                blocks=compile_result.blocks,
                plan=plan,
                report=report,
            )
        )

    telemetry_summary = summarize_telemetry(tuple(telemetry_records)).to_dict()
    provider_summary = _summarize_provider_trace(source_records)
    shadow_summary = _summarize_shadow_trial_trace(source_records)
    summary = {
        "schema_version": "prefix-reorder-dataset-eval-summary-v1",
        "input_path": str(source_path),
        "input_record_count": len(source_records),
        "supported_request_count": len(telemetry_records),
        "unsupported_record_count": sum(skipped_reason_counts.values()),
        "semantic_coverage_supported": bool(telemetry_records),
        "natural_language_segmentation_enabled": enable_natural_language_segmentation,
        "groupchat_history_reordering_enabled": enable_groupchat_history_reordering,
        "skipped_reason_counts": dict(sorted(skipped_reason_counts.items())),
        "semantic_type_counts": dict(sorted(semantic_type_counts.items())),
        "movability_counts": dict(sorted(movability_counts.items())),
        "share_scope_counts": dict(sorted(share_scope_counts.items())),
        "moved_semantic_type_counts": dict(sorted(moved_semantic_type_counts.items())),
        "rule_gap_diagnostics": _summarize_rule_gap_diagnostics(
            repeated_system_blocks,
            natural_language_hint_counts,
            semantic_candidates_by_parent,
        ),
        "provider_trace": provider_summary,
        "shadow_trial_trace": shadow_summary,
        **telemetry_summary,
    }

    telemetry_target = _write_jsonl(telemetry_path, telemetry_records)
    summary_target = _write_json(summary_path, summary)
    return DatasetEvaluationResult(
        input_path=str(source_path),
        telemetry_path=str(telemetry_target) if telemetry_target else None,
        summary_path=str(summary_target) if summary_target else None,
        summary=summary,
    )


def openai_messages_to_autogen(messages: Any) -> tuple[LLMMessage, ...]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    converted: list[LLMMessage] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"message {index} is not an object")

        message_type = str(message.get("type") or "")
        if message_type in {
            "SystemMessage",
            "UserMessage",
            "AssistantMessage",
            "FunctionExecutionResultMessage",
        }:
            converted.append(_autogen_dump_to_message(message, index=index))
            continue

        role = str(message.get("role") or "user").lower()
        source = str(message.get("name") or role or "unknown")
        content = _content_to_text(message.get("content"))

        if role in {"system", "developer"}:
            converted.append(SystemMessage(content=content))
        elif role == "user":
            converted.append(UserMessage(content=content, source=source))
        elif role == "assistant":
            converted.append(AssistantMessage(content=content, source=source))
        elif role in {"tool", "function"}:
            converted.append(
                FunctionExecutionResultMessage(
                    content=[
                        FunctionExecutionResult(
                            content=content,
                            name=str(message.get("name") or role),
                            call_id=str(message.get("tool_call_id") or message.get("call_id") or f"call-{index}"),
                        )
                    ]
                )
            )
        else:
            converted.append(UserMessage(content=content, source=source))

    return tuple(converted)


def _autogen_dump_to_message(message: Mapping[str, Any], *, index: int) -> LLMMessage:
    message_type = str(message.get("type") or "")
    content = message.get("content")
    if message_type == "SystemMessage":
        return SystemMessage(content=_content_to_text(content))
    if message_type == "UserMessage":
        return UserMessage(content=_content_to_text(content), source=str(message.get("source") or "user"))
    if message_type == "AssistantMessage":
        return AssistantMessage(content=_content_to_text(content), source=str(message.get("source") or "assistant"))
    if message_type == "FunctionExecutionResultMessage":
        results: list[FunctionExecutionResult] = []
        if isinstance(content, list):
            for result_index, item in enumerate(content):
                if isinstance(item, Mapping):
                    results.append(
                        FunctionExecutionResult(
                            content=_content_to_text(item.get("content")),
                            name=str(item.get("name") or "tool"),
                            call_id=str(item.get("call_id") or f"call-{index}-{result_index}"),
                            is_error=item.get("is_error") if isinstance(item.get("is_error"), bool) else None,
                        )
                    )
                else:
                    results.append(
                        FunctionExecutionResult(
                            content=_content_to_text(item),
                            name="tool",
                            call_id=f"call-{index}-{result_index}",
                        )
                    )
        if not results:
            results.append(FunctionExecutionResult(content="", name="tool", call_id=f"call-{index}"))
        return FunctionExecutionResultMessage(content=results)
    raise ValueError(f"unsupported AutoGen message type: {message_type}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate prefix-reorder rule coverage on JSONL chat requests or provider traces."
    )
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--telemetry", help="Optional prompt-safe per-request telemetry JSONL output path.")
    parser.add_argument("--summary", help="Optional summary JSON output path.")
    parser.add_argument("--session-id", default="dataset-eval", help="Default planner session id.")
    parser.add_argument("--max-records", type=int, help="Optional maximum records to read.")
    parser.add_argument(
        "--enable-natural-language-segmentation",
        action="store_true",
        help="Opt-in offline analysis mode: split unmarked system prompts by line and classify low-risk policy/tool lines.",
    )
    parser.add_argument(
        "--enable-groupchat-history-reordering",
        action="store_true",
        help="Opt-in offline analysis mode: move exact repeated groupchat dialogue prefix ahead of agent-local suffix.",
    )
    parser.add_argument(
        "--require-messages",
        action="store_true",
        help="Return a non-zero exit code when no records contain structured chat messages.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_dataset(
        input_path=args.input,
        telemetry_path=args.telemetry,
        summary_path=args.summary,
        session_id=args.session_id,
        max_records=args.max_records,
        enable_natural_language_segmentation=args.enable_natural_language_segmentation,
        enable_groupchat_history_reordering=args.enable_groupchat_history_reordering,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    if args.require_messages and not result.summary["semantic_coverage_supported"]:
        return 2
    return 0


def _load_jsonl(path: Path, *, max_records: int | None) -> Iterable[dict[str, Any]]:
    count = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                continue
            yield value
            count += 1
            if max_records is not None and count >= max_records:
                break


def _extract_openai_request(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for candidate in _candidate_mappings(record):
        messages = candidate.get("messages")
        if isinstance(messages, list):
            return candidate
    return None


def _candidate_mappings(record: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    yield record
    for key in ("body", "request_body", "request", "payload", "input", "json"):
        value = record.get(key)
        if isinstance(value, Mapping):
            yield value
        elif isinstance(value, str) and value.strip().startswith("{"):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, Mapping):
                yield parsed


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, Mapping):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


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


def _build_eval_record(
    *,
    record: Mapping[str, Any],
    record_index: int,
    effective_session_id: str,
    messages: Sequence[LLMMessage],
    blocks: Sequence[PromptBlock],
    plan: Any,
    report: Any,
) -> dict[str, Any]:
    blocks_by_id = {block.block_id: block for block in blocks}
    moved_details = tuple(
        _block_telemetry(blocks_by_id[block_id]) for block_id in plan.moved_blocks if block_id in blocks_by_id
    )
    return {
        "schema_version": "prefix-reorder-dataset-eval-v1",
        "source_record_index": record_index,
        "source_record_id": record.get("id") or record.get("request_id"),
        "session_id": effective_session_id,
        "message_count": len(messages),
        "message_types": tuple(getattr(message, "type", type(message).__name__) for message in messages),
        "block_count": len(blocks),
        "semantic_type_counts": dict(Counter(block.semantic_type.value for block in blocks)),
        "movability_counts": dict(Counter(block.movability.value for block in blocks)),
        "share_scope_counts": dict(Counter(block.share_scope.value for block in blocks)),
        "blocks_moved": plan.moved_blocks,
        "moved_block_details": moved_details,
        "cacheable_prefix_blocks": plan.cacheable_prefix_blocks,
        "prefix_tree": serialize_prefix_tree(plan.prefix_tree),
        "validation": {
            "applied": report.applied,
            "fallback": report.fallback,
            "reason": report.reason,
        },
        "utility_estimate": dataclass_to_dict(report.utility_estimate),
        "semantic_guard": dataclass_to_dict(report.semantic_guard_report),
        "risk_notes": report.risk_notes,
    }


def _block_telemetry(block: PromptBlock) -> dict[str, Any]:
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


def _observe_rule_gap_candidate(
    repeated_system_blocks: dict[str, dict[str, Any]],
    natural_language_hint_counts: Counter[str],
    semantic_candidates_by_parent: dict[str, tuple[SemanticCandidate, ...]],
    *,
    block: PromptBlock,
    record_index: int,
) -> None:
    if not block.is_system_text or not isinstance(block.rendered_text, str):
        return
    if block.share_scope in {ShareScope.GLOBAL, ShareScope.SUBGROUP}:
        return
    if block.movability not in {Movability.LOCAL_ONLY, Movability.ORDER_SENSITIVE}:
        return

    hints = _natural_language_shared_prefix_hints(block.rendered_text)
    for hint in hints:
        natural_language_hint_counts[hint] += 1
    semantic_candidates_by_parent.setdefault(
        block.content_hash,
        extract_semantic_candidates(block.rendered_text, parent_hash=block.content_hash),
    )

    record = repeated_system_blocks.setdefault(
        block.content_hash,
        {
            "content_hash": block.content_hash,
            "count": 0,
            "first_record_index": record_index,
            "semantic_type": block.semantic_type.value,
            "movability": block.movability.value,
            "share_scope": block.share_scope.value,
            "char_count": len(block.rendered_text),
            "natural_language_shared_prefix_hints": sorted(hints),
        },
    )
    record["count"] += 1


def _summarize_rule_gap_diagnostics(
    repeated_system_blocks: Mapping[str, Mapping[str, Any]],
    natural_language_hint_counts: Counter[str],
    semantic_candidates_by_parent: Mapping[str, tuple[SemanticCandidate, ...]],
) -> dict[str, Any]:
    repeated = [
        dict(record)
        for record in repeated_system_blocks.values()
        if isinstance(record.get("count"), int) and record["count"] >= 2
    ]
    repeated_long = [record for record in repeated if int(record.get("char_count") or 0) >= 80]
    repeated.sort(key=lambda record: (-int(record["count"]), str(record["content_hash"])))
    repeated_long.sort(key=lambda record: (-int(record["count"]), -int(record.get("char_count") or 0)))
    hinted = [record for record in repeated_long if record.get("natural_language_shared_prefix_hints")]
    repeated_hashes = {str(record["content_hash"]) for record in repeated_long}
    repeated_candidates = tuple(
        candidate
        for parent_hash in repeated_hashes
        for candidate in semantic_candidates_by_parent.get(parent_hash, ())
    )
    all_candidates = tuple(
        candidate
        for candidates in semantic_candidates_by_parent.values()
        for candidate in candidates
    )
    return {
        "repeated_nonprefix_system_block_count": len(repeated),
        "repeated_nonprefix_system_block_occurrences": sum(int(record["count"]) for record in repeated),
        "repeated_long_nonprefix_system_block_count": len(repeated_long),
        "natural_language_hint_counts": dict(sorted(natural_language_hint_counts.items())),
        "repeated_natural_language_hint_block_count": len(hinted),
        "all_nonprefix_semantic_candidate_diagnostics": semantic_candidate_summary_to_dict(
            summarize_semantic_candidates(all_candidates)
        ),
        "semantic_candidate_diagnostics": semantic_candidate_summary_to_dict(
            summarize_semantic_candidates(repeated_candidates)
        ),
        "top_repeated_nonprefix_system_blocks": repeated_long[:10],
    }


def _natural_language_shared_prefix_hints(text: str) -> tuple[str, ...]:
    hints: set[str] = set()
    lowered = text.lower()
    normalized_lines = [
        re.sub(r"\s+", " ", line.strip())
        for line in text.splitlines()
        if line.strip()
    ]

    if "do not" in lowered or "don't" in lowered or "must" in lowered or "cannot" in lowered:
        hints.add("policy_like_instruction")
    if "tool" in lowered or "code block" in lowered or "execute" in lowered or "script" in lowered:
        hints.add("tool_or_code_policy")
    if "verify" in lowered or "evidence" in lowered or "check" in lowered:
        hints.add("verification_policy")
    if len(normalized_lines) >= 3 and any(line[:2].isdigit() or line.startswith(("-", "*")) for line in normalized_lines):
        hints.add("structured_procedure")
    if len(text) >= 400 and len(normalized_lines) >= 4:
        hints.add("long_stable_system_instruction")
    return tuple(sorted(hints))


def _summarize_provider_trace(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    provider_records = [
        record
        for record in records
        if any(
            key in record
            for key in (
                "actual_prompt_tokens",
                "actual_cached_tokens",
                "latency_seconds",
                "latency_ms",
                "actual_cost_usd",
                "cost_usd",
                "total_cost_usd",
                "usage",
                "transformation_applied",
                "rewrite_applied",
                "upstream_status",
            )
        )
    ]
    actual_prompt_tokens = sum(_provider_prompt_tokens(record) for record in provider_records)
    actual_cached_tokens = sum(_provider_cached_tokens(record) for record in provider_records)
    actual_completion_tokens = sum(_provider_completion_tokens(record) for record in provider_records)
    actual_total_tokens = sum(_provider_total_tokens(record) for record in provider_records)
    local_input_tokens = sum(_numeric(record.get("local_input_tokens")) for record in provider_records)
    estimated_cached_tokens = sum(_numeric(record.get("estimated_cached_tokens")) for record in provider_records)
    latency_values = [
        value
        for record in provider_records
        for value in (_latency_seconds(record),)
        if value is not None
    ]
    cost_values = [
        value
        for record in provider_records
        for value in (_cost_usd(record),)
        if value is not None
    ]
    actual_cost_usd = sum(cost_values)
    return {
        "supported": bool(provider_records),
        "record_count": len(provider_records),
        "ok_count": sum(1 for record in provider_records if record.get("error") in {None, ""}),
        "error_count": sum(1 for record in provider_records if record.get("error") not in {None, ""}),
        "transformed_count": sum(
            1 for record in provider_records if bool(record.get("transformation_applied") or record.get("rewrite_applied"))
        ),
        "actual_prompt_tokens": actual_prompt_tokens,
        "actual_cached_tokens": actual_cached_tokens,
        "actual_completion_tokens": actual_completion_tokens,
        "actual_total_tokens": actual_total_tokens,
        "actual_cache_hit_ratio": _ratio(actual_cached_tokens, actual_prompt_tokens),
        "local_input_tokens": local_input_tokens,
        "estimated_cached_tokens": estimated_cached_tokens,
        "estimated_cache_hit_ratio": _ratio(estimated_cached_tokens, local_input_tokens),
        "cost_supported": bool(cost_values),
        "actual_cost_usd": actual_cost_usd,
        "average_cost_usd": _ratio(actual_cost_usd, len(cost_values)),
        "latency_supported": bool(latency_values),
        "average_latency_seconds": _ratio(sum(latency_values), len(latency_values)),
        "p50_latency_seconds": _percentile(latency_values, 50),
        "p95_latency_seconds": _percentile(latency_values, 95),
        "p99_latency_seconds": _percentile(latency_values, 99),
    }


def _summarize_shadow_trial_trace(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    shadow_records = [
        record.get("shadow_trial")
        for record in records
        if isinstance(record.get("shadow_trial"), Mapping)
    ]
    action_counts: Counter[str] = Counter()
    review_status_counts: Counter[str] = Counter()
    matched_rule_ids: set[str] = set()
    for shadow in shadow_records:
        action_counts.update(
            {
                str(action): _numeric(count)
                for action, count in (
                    shadow.get("trial_action_counts")
                    if isinstance(shadow.get("trial_action_counts"), Mapping)
                    else {}
                ).items()
            }
        )
        rows = shadow.get("matched_rules") if isinstance(shadow.get("matched_rules"), list) else []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            rule_id = row.get("trial_rule_id")
            if rule_id:
                matched_rule_ids.add(str(rule_id))
            status = row.get("review_status")
            if status:
                review_status_counts[str(status)] += 1
    matched_counts = [_numeric(shadow.get("matched_rule_count")) for shadow in shadow_records]
    unsupported_counts = [_numeric(shadow.get("unsupported_rule_count")) for shadow in shadow_records]
    return {
        "schema_version": "prefix-shadow-trial-trace-summary-v1",
        "supported": bool(shadow_records),
        "record_count": len(shadow_records),
        "record_with_match_count": sum(1 for count in matched_counts if count > 0),
        "matched_rule_observation_count": sum(matched_counts),
        "unique_matched_rule_count": len(matched_rule_ids),
        "unique_matched_rule_ids": tuple(sorted(matched_rule_ids)),
        "trial_action_counts": dict(sorted(action_counts.items())),
        "review_status_counts": dict(sorted(review_status_counts.items())),
        "unsupported_rule_observation_count": sum(unsupported_counts),
        "validator_behavior_change_allowed": False,
        "safe_for_automatic_validator_promotion": False,
        "automation_policy": "shadow_only_no_behavior_change",
    }


def _numeric(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return 0
    return 0


def _provider_prompt_tokens(record: Mapping[str, Any]) -> int:
    direct = _numeric(record.get("actual_prompt_tokens"))
    if direct:
        return direct
    usage = record.get("usage")
    if isinstance(usage, Mapping):
        return _numeric(usage.get("prompt_tokens") or usage.get("input_tokens"))
    return 0


def _provider_completion_tokens(record: Mapping[str, Any]) -> int:
    direct = _numeric(record.get("actual_completion_tokens"))
    if direct:
        return direct
    usage = record.get("usage")
    if isinstance(usage, Mapping):
        return _numeric(usage.get("completion_tokens") or usage.get("output_tokens"))
    return 0


def _provider_total_tokens(record: Mapping[str, Any]) -> int:
    direct = _numeric(record.get("actual_total_tokens"))
    if direct:
        return direct
    usage = record.get("usage")
    if isinstance(usage, Mapping):
        total = _numeric(usage.get("total_tokens"))
        if total:
            return total
    return _provider_prompt_tokens(record) + _provider_completion_tokens(record)


def _provider_cached_tokens(record: Mapping[str, Any]) -> int:
    direct = _numeric(record.get("actual_cached_tokens"))
    if direct:
        return direct
    usage = record.get("usage")
    if not isinstance(usage, Mapping):
        return 0
    for key in (
        "cached_tokens",
        "cached_prompt_tokens",
        "prompt_cache_hit_tokens",
        "cache_hit_tokens",
        "input_cached_tokens",
    ):
        value = _numeric(usage.get(key))
        if value:
            return value
    for detail_key in ("prompt_tokens_details", "input_token_details", "input_tokens_details"):
        details = usage.get(detail_key)
        if isinstance(details, Mapping):
            value = _provider_cached_tokens_from_details(details)
            if value:
                return value
    return 0


def _provider_cached_tokens_from_details(details: Mapping[str, Any]) -> int:
    for key in (
        "cached_tokens",
        "cache_read",
        "cached",
        "cache_hit_tokens",
        "prompt_cache_hit_tokens",
    ):
        value = _numeric(details.get(key))
        if value:
            return value
    return 0


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _latency_seconds(record: Mapping[str, Any]) -> float | None:
    seconds = _float_value(record.get("latency_seconds"))
    if seconds is not None:
        return seconds
    milliseconds = _float_value(record.get("latency_ms"))
    if milliseconds is not None:
        return milliseconds / 1000.0
    return None


def _cost_usd(record: Mapping[str, Any]) -> float | None:
    for key in ("actual_cost_usd", "cost_usd", "total_cost_usd"):
        value = _float_value(record.get(key))
        if value is not None:
            return value
    usage = record.get("usage")
    if isinstance(usage, Mapping):
        for key in ("cost_usd", "cost", "total_cost_usd", "total_cost"):
            value = _float_value(usage.get(key))
            if value is not None:
                return value
    return None


def _ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _write_jsonl(path: str | Path | None, records: Sequence[Mapping[str, Any]]) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def _write_json(path: str | Path | None, value: Mapping[str, Any]) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return target


if __name__ == "__main__":
    raise SystemExit(main())
