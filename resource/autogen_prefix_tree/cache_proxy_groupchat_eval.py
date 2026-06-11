from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .openai_request_adapter import OpenAICompatibleRequestAdapter
from .project_api_config import load_project_provider_config
from .real_groupchat_replay_ab import _prepare_body


ROOT_DIR = Path(__file__).resolve().parents[1]
CACHE_PROXY_DIR = ROOT_DIR / "cache_hit_proxy"

sys.path.append(str(CACHE_PROXY_DIR))

from cache_estimator import estimate_cache_hit  # noqa: E402
from request_recorder import RequestRecorder  # noqa: E402


def run_cache_proxy_groupchat_eval(
    *,
    capture_path: str | Path,
    output: str | Path,
    max_records: int = 3,
    max_tokens: int = 10000,
    run_id: str = "cache-proxy",
    block_size: int = 64,
    include_ideal: bool = True,
    input_token_source: str = "prompt_boundary",
) -> dict[str, Any]:
    rows = _load_capture_rows(Path(capture_path), max_records=max_records)
    config = load_project_provider_config()
    model = config.model if config is not None else "deepseek-v4-pro"
    variants: list[str] = ["baseline", "plugin"]
    if include_ideal:
        variants.append("ideal_upper_bound")

    all_rows: list[dict[str, Any]] = []
    for variant in variants:
        recorder = _make_recorder(block_size=block_size)
        adapter = OpenAICompatibleRequestAdapter(
            session_id=f"cache-proxy-groupchat-{variant}",
            enabled=variant == "plugin",
            enable_groupchat_history_reordering=variant == "plugin",
        )
        for request_index, source in enumerate(rows, start=1):
            original_body = _prepare_body(
                source["body"],
                model=model,
                max_tokens=max_tokens,
                variant=variant,
                run_id=run_id,
            )
            if variant == "baseline":
                rewritten_body = original_body
                rewrite_applied = False
                rewrite_reason = "disabled"
                blocks_moved = 0
            elif variant == "plugin":
                rewrite = adapter.rewrite_request_body(original_body)
                rewritten_body = rewrite.rewritten_body
                rewrite_applied = rewrite.applied
                rewrite_reason = (rewrite.telemetry.get("validation") or {}).get("reason")
                blocks_moved = len(rewrite.telemetry.get("blocks_moved") or ())
            else:
                rewritten_body = _build_ideal_shared_prefix_body(original_body, str(source.get("agent_name") or ""))
                rewrite_applied = True
                rewrite_reason = "offline_ideal_shared_prefix"
                blocks_moved = 0

            request_record = recorder.create_request_record(
                session_id=f"cache-proxy-groupchat-{variant}",
                request_id=f"{variant}-{request_index}-{source.get('agent_name')}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                request_body=rewritten_body,
                request_body_bytes=_stable_request_bytes(rewritten_body),
                conversation_mode="piai_probe",
            )
            if input_token_source == "raw_body_lcp":
                request_record = recorder.apply_input_token_source(request_record, "openclaw_raw_body")
            history = recorder.history_snapshot()
            if input_token_source == "prompt_lcp":
                estimate = _estimate_prompt_lcp(request_record, history)
            else:
                estimate = estimate_cache_hit(request_record, history)
            recorder.touch_request(estimate.get("matched_request_id"))
            recorder.append_history(request_record)

            all_rows.append(
                {
                    "variant": variant,
                    "request_index": request_index,
                    "agent": source.get("agent_name"),
                    "rewrite_applied": rewrite_applied,
                    "rewrite_reason": rewrite_reason,
                    "blocks_moved": blocks_moved,
                    "predicted_input_tokens": int(request_record.get("local_input_tokens") or 0),
                    "estimated_cached_tokens": int(estimate.get("estimated_cached_tokens") or 0),
                    "estimated_cache_hit_rate": float(estimate.get("estimated_cache_hit_rate") or 0.0),
                    "matched_request_id": estimate.get("matched_request_id"),
                    "match_strategy": estimate.get("match_strategy"),
                    "estimation_denominator_tokens": int(estimate.get("estimation_denominator_tokens") or 0),
                    "message_shape": _message_shape(rewritten_body),
                }
            )

    summary = {
        "schema_version": "cache-proxy-autogen-groupchat-eval-v1",
        "capture_path": str(capture_path),
        "source": "local cache_hit_proxy estimate over real AutoGen RoundRobin captured request bodies",
        "model": model,
        "run_id": run_id,
        "block_size": block_size,
        "input_token_source": input_token_source,
        "request_count_per_variant": len(rows),
        "variants": {
            variant: _variant_summary([row for row in all_rows if row["variant"] == variant])
            for variant in variants
        },
        "delta_vs_baseline": {},
        "rows": all_rows,
    }
    baseline = summary["variants"]["baseline"]
    summary["delta_vs_baseline"] = {
        variant: _delta(baseline, metrics)
        for variant, metrics in summary["variants"].items()
        if variant != "baseline"
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(output_path.with_suffix(".md"), summary)
    return summary


def _load_capture_rows(path: Path, *, max_records: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    agent_rows = [row for row in rows if row.get("request_type") in {None, "agent"} and isinstance(row.get("body"), Mapping)]
    return agent_rows[:max_records]


def _make_recorder(*, block_size: int) -> RequestRecorder:
    return RequestRecorder(
        traces_dir=str(ROOT_DIR / "autogen_groupchat_kvcache_experiment" / "results" / "_cache_proxy_eval_traces"),
        tokenizer_dir=str(CACHE_PROXY_DIR / "deepseek_tokenizer"),
        tokenizer_preset="deepseek-v4-pro",
        block_size=block_size,
        cache_idle_ttl_hours=24,
    )


def _stable_request_bytes(body: Mapping[str, Any]) -> bytes:
    return json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _estimate_prompt_lcp(current_request: Mapping[str, Any], history_requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    current_tokens = current_request.get("token_ids", [])
    if not isinstance(current_tokens, list):
        current_tokens = []
    current_model = current_request.get("model")
    block_size = int(current_request.get("cache_block_size") or 64)
    denominator = int(current_request.get("cache_estimation_input_tokens") or len(current_tokens) or 0)
    best = 0
    matched = None
    for history_item in history_requests:
        if history_item.get("model") != current_model:
            continue
        history_tokens = history_item.get("token_ids", [])
        if not isinstance(history_tokens, list):
            continue
        candidate = _snap_down(_common_prefix_len(current_tokens, history_tokens), block_size)
        if candidate > best:
            best = candidate
            matched = history_item.get("request_id")
    if best == 0:
        matched = None
    return {
        "estimated_cached_tokens": best,
        "estimated_cache_hit_rate": best / denominator if denominator else 0.0,
        "matched_request_id": matched,
        "match_strategy": "deepseek_prompt_lcp_block_aligned",
        "estimation_denominator_tokens": denominator,
        "openclaw_session_cache_floor_tokens": 0,
        "openclaw_global_cache_floor_tokens": 0,
    }


def _common_prefix_len(left: Sequence[Any], right: Sequence[Any]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _snap_down(token_count: int, block_size: int) -> int:
    if block_size <= 1:
        return max(token_count, 0)
    if token_count < block_size:
        return 0
    return token_count - (token_count % block_size)


def _message_shape(body: Mapping[str, Any]) -> str:
    parts: list[str] = []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return "messages=<non-list>"
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content", "")
        content_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        name = message.get("name")
        name_part = f":{name}" if name else ""
        parts.append(f"{message.get('role', 'unknown')}{name_part}:{len(content_text)}")
    return " ; ".join(parts)


def _variant_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prompt_tokens = sum(int(row.get("predicted_input_tokens") or 0) for row in rows)
    cached_tokens = sum(int(row.get("estimated_cached_tokens") or 0) for row in rows)
    return {
        "request_count": len(rows),
        "rewritten_count": sum(1 for row in rows if row.get("rewrite_applied") is True),
        "predicted_input_tokens": prompt_tokens,
        "estimated_cached_tokens": cached_tokens,
        "estimated_cache_hit_ratio": cached_tokens / prompt_tokens if prompt_tokens else 0.0,
        "warm_predicted_input_tokens": sum(
            int(row.get("predicted_input_tokens") or 0) for row in rows if int(row.get("request_index") or 0) > 1
        ),
        "warm_estimated_cached_tokens": sum(
            int(row.get("estimated_cached_tokens") or 0) for row in rows if int(row.get("request_index") or 0) > 1
        ),
    }


def _delta(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    baseline_tokens = int(baseline.get("predicted_input_tokens") or 0)
    candidate_tokens = int(candidate.get("predicted_input_tokens") or 0)
    baseline_cached = int(baseline.get("estimated_cached_tokens") or 0)
    candidate_cached = int(candidate.get("estimated_cached_tokens") or 0)
    return {
        "estimated_cached_tokens_delta": candidate_cached - baseline_cached,
        "estimated_cache_hit_ratio_delta": float(candidate.get("estimated_cache_hit_ratio") or 0.0)
        - float(baseline.get("estimated_cache_hit_ratio") or 0.0),
        "predicted_input_tokens_delta": candidate_tokens - baseline_tokens,
    }


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    variants = summary["variants"]
    lines = [
        "# Cache Proxy AutoGen RoundRobin Estimate",
        "",
        f"Source: `{summary['capture_path']}`",
        "",
        "| Variant | Requests | Rewritten | Input tokens | Estimated cached | Estimated hit ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, metrics in variants.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(variant),
                    str(metrics["request_count"]),
                    str(metrics["rewritten_count"]),
                    str(metrics["predicted_input_tokens"]),
                    str(metrics["estimated_cached_tokens"]),
                    f"{float(metrics['estimated_cache_hit_ratio']):.4%}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Per request:",
            "",
            "| Variant | Agent | Rewrite | Input tokens | Estimated cached | Hit ratio | Match | Shape |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    str(row["agent"]),
                    str(row["rewrite_applied"]).lower(),
                    str(row["predicted_input_tokens"]),
                    str(row["estimated_cached_tokens"]),
                    f"{float(row['estimated_cache_hit_rate']):.4%}",
                    str(row.get("matched_request_id") or ""),
                    str(row.get("message_shape") or ""),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_ideal_shared_prefix_body(body: Mapping[str, Any], agent: str) -> dict[str, Any]:
    original = copy.deepcopy(dict(body))
    system_content = _first_system_content(original)
    role_instruction = _marker_block(
        system_content,
        "ROLE_SPECIFIC_INSTRUCTION_START",
        "ROLE_SPECIFIC_INSTRUCTION_END",
    )
    user_task = _marker_block(system_content, "USER_TASK_START", "USER_TASK_END")
    shared_context = _marker_blocks(
        system_content,
        "SHARED_GROUPCHAT_CONTEXT_START",
        "SHARED_GROUPCHAT_CONTEXT_END",
    )
    tool_schema = _marker_block(system_content, "TOOL_SCHEMA_START", "TOOL_SCHEMA_END")
    current_turn = _marker_block(
        system_content,
        "CURRENT_TURN_INSTRUCTION_START",
        "CURRENT_TURN_INSTRUCTION_END",
    )
    history_chunks: list[str] = []
    for message in original.get("messages") or []:
        if not isinstance(message, Mapping) or message.get("role") == "system":
            continue
        content = message.get("content", "")
        content_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        if "USER_TASK_START" in content_text and "USER_TASK_END" in content_text:
            continue
        history_chunks.append(
            "GROUP_CHAT_HISTORY_MESSAGE_START\n"
            f"role={message.get('role', 'unknown')}\n"
            f"{content_text}\n"
            "GROUP_CHAT_HISTORY_MESSAGE_END"
        )
    history_block = (
        "GROUP_CHAT_HISTORY_PREFIX_START\n"
        + "\n\n".join(history_chunks)
        + "\nGROUP_CHAT_HISTORY_PREFIX_END"
        if history_chunks
        else "GROUP_CHAT_HISTORY_PREFIX_START\n<empty>\nGROUP_CHAT_HISTORY_PREFIX_END"
    )
    ideal_system = "\n\n".join(
        part
        for part in [
            "IDEAL_SHARED_PREFIX_LAYOUT_START",
            user_task,
            shared_context,
            tool_schema,
            history_block,
            current_turn,
            "IDEAL_SHARED_PREFIX_LAYOUT_END",
            role_instruction,
            (
                "IDEAL_LAYOUT_NOTE_START\n"
                f"Offline upper-bound layout for agent={agent}.\n"
                "IDEAL_LAYOUT_NOTE_END"
            ),
        ]
        if part
    )
    original["messages"] = [{"role": "system", "content": ideal_system}]
    original.pop("tools", None)
    return original


def _first_system_content(body: Mapping[str, Any]) -> str:
    for message in body.get("messages") or []:
        if isinstance(message, Mapping) and message.get("role") == "system":
            content = message.get("content", "")
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return ""


def _marker_block(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    end = text.find(end_marker, start)
    if end < 0:
        return ""
    return text[start : end + len(end_marker)]


def _marker_blocks(text: str, start_marker: str, end_marker: str) -> str:
    blocks: list[str] = []
    pos = 0
    while True:
        start = text.find(start_marker, pos)
        if start < 0:
            break
        end = text.find(end_marker, start)
        if end < 0:
            break
        end += len(end_marker)
        blocks.append(text[start:end])
        pos = end
    return "\n\n".join(blocks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate real AutoGen GroupChat cache hits with local cache_hit_proxy.")
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--run-id", default="cache-proxy")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--no-ideal", action="store_true")
    parser.add_argument(
        "--input-token-source",
        choices=("prompt_boundary", "prompt_lcp", "raw_body_lcp"),
        default="prompt_boundary",
        help=(
            "prompt_boundary uses cache_hit_proxy default boundary-prefix units; "
            "prompt_lcp uses the same proxy tokenized prompt stream with block-aligned LCP; "
            "raw_body_lcp uses its raw-body LCP path."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_cache_proxy_groupchat_eval(
        capture_path=args.capture,
        output=args.output,
        max_records=args.max_records,
        max_tokens=args.max_tokens,
        run_id=args.run_id,
        block_size=args.block_size,
        include_ideal=not args.no_ideal,
        input_token_source=args.input_token_source,
    )
    print(json.dumps({"variants": summary["variants"], "delta": summary["delta_vs_baseline"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
