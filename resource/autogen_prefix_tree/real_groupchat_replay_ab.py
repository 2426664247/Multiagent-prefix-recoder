from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from .openai_request_adapter import OpenAICompatibleRequestAdapter
from .project_api_config import DEFAULT_DEEPSEEK_BASE_URL, load_project_provider_config


def run_replay_ab(
    *,
    capture_path: str | Path,
    output: str | Path,
    max_records: int = 3,
    max_tokens: int = 96,
    timeout_seconds: float = 120.0,
    run_id: str | None = None,
) -> dict[str, Any]:
    config = load_project_provider_config()
    if config is None:
        raise RuntimeError("Project DeepSeek config is missing")
    effective_run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
    source_rows = _load_capture_rows(Path(capture_path), max_records=max_records)
    variants = {
        "baseline": OpenAICompatibleRequestAdapter(session_id="real-groupchat-replay-baseline", enabled=False),
        "plugin": OpenAICompatibleRequestAdapter(
            session_id="real-groupchat-replay-plugin",
            enable_groupchat_history_reordering=True,
        ),
    }
    rows: list[dict[str, Any]] = []
    for variant, adapter in variants.items():
        for request_index, source in enumerate(source_rows, start=1):
            original_body = _prepare_body(
                source["body"],
                model=config.model,
                max_tokens=max_tokens,
                variant=variant,
                run_id=effective_run_id,
            )
            rewrite = adapter.rewrite_request_body(original_body)
            started = time.perf_counter()
            response = _post_chat_completion(
                base_url=config.base_url or DEFAULT_DEEPSEEK_BASE_URL,
                api_key=config.api_key,
                body=rewrite.rewritten_body,
                timeout_seconds=timeout_seconds,
            )
            latency_seconds = time.perf_counter() - started
            content = _response_content(response)
            message = _response_message(response)
            usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
            row = {
                "variant": variant,
                "request_index": request_index,
                "agent": source.get("agent_name"),
                "rewrite_applied": rewrite.applied,
                "rewrite_reason": (rewrite.telemetry.get("validation") or {}).get("reason"),
                "blocks_moved": len(rewrite.telemetry.get("blocks_moved") or ()),
                "prompt_tokens": _int(usage.get("prompt_tokens")),
                "completion_tokens": _int(usage.get("completion_tokens")),
                "total_tokens": _int(usage.get("total_tokens")),
                "cached_tokens": _cached_tokens(usage),
                "latency_seconds": latency_seconds,
                "utility_ok": _utility_ok(content, source.get("agent_name")),
                "content_preview": content[:240],
                "response_message": message,
            }
            rows.append(row)

    summary = {
        "schema_version": "real-autogen-groupchat-replay-ab-v1",
        "capture_path": str(capture_path),
        "source": "replay of real AutoGen RoundRobinGroupChat OpenAI-compatible captured worker request bodies",
        "model": config.model,
        "run_id": effective_run_id,
        "request_count_per_variant": len(source_rows),
        "variants": {
            variant: _variant_summary([row for row in rows if row["variant"] == variant])
            for variant in variants
        },
        "delta": {},
        "rows": rows,
    }
    summary["delta"] = _delta(summary["variants"]["baseline"], summary["variants"]["plugin"])
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _load_capture_rows(path: Path, *, max_records: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    agent_rows = [row for row in rows if row.get("request_type") in {None, "agent"} and isinstance(row.get("body"), Mapping)]
    return agent_rows[:max_records]


def _prepare_body(body: Mapping[str, Any], *, model: str, max_tokens: int, variant: str, run_id: str) -> dict[str, Any]:
    prepared = dict(body)
    prepared["model"] = model
    prepared["temperature"] = 0
    prepared["stream"] = False
    prepared["max_tokens"] = max_tokens
    prepared["user"] = f"real-autogen-groupchat-{variant}-{run_id}"
    prepared["response_format"] = {"type": "json_object"}
    messages = [dict(message) for message in prepared.get("messages") or () if isinstance(message, Mapping)]
    if messages:
        first = dict(messages[0])
        first["content"] = _inject_variant_marker(str(first.get("content") or ""), variant=variant, run_id=run_id)
        first["content"] = _replace_current_turn_instruction(str(first.get("content") or ""))
        messages[0] = first
    prepared["messages"] = messages
    return prepared


def _post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    body: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _response_content(response: Mapping[str, Any]) -> str:
    message = _response_message(response)
    return str(message.get("content") or "") if isinstance(message, Mapping) else ""


def _response_message(response: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else {}
    return message if isinstance(message, Mapping) else {}


def _inject_variant_marker(content: str, *, variant: str, run_id: str) -> str:
    marker = f"AB_REPLAY_VARIANT_ID: {variant}-{run_id}"
    if marker in content:
        return content
    if "USER_TASK_START\n" in content:
        return content.replace("USER_TASK_START\n", f"USER_TASK_START\n{marker}\n", 1)
    return f"{marker}\n{content}"


def _replace_current_turn_instruction(content: str) -> str:
    instruction = (
        "CURRENT_TURN_INSTRUCTION_START\n"
        "Return only JSON with keys utility_pass, agent, checksum, final_answer. "
        "Set utility_pass=true, checksum=1369, and agent to your AGENT_NAME. "
        "No markdown, no prose.\n"
        "CURRENT_TURN_INSTRUCTION_END"
    )
    start = content.find("CURRENT_TURN_INSTRUCTION_START")
    end_marker = "CURRENT_TURN_INSTRUCTION_END"
    end = content.find(end_marker, start)
    if start >= 0 and end >= 0:
        return content[:start] + instruction + content[end + len(end_marker) :]
    return content + "\n\n" + instruction


def _utility_ok(content: str, agent: Any) -> bool:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return False
    return (
        parsed.get("utility_pass") is True
        and _int(parsed.get("checksum")) == 1369
        and str(parsed.get("agent") or "").strip().lower() == str(agent or "").strip().lower()
    )


def _cached_tokens(usage: Mapping[str, Any]) -> int:
    details = usage.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        return _int(details.get("cached_tokens") or details.get("cache_read") or details.get("cached"))
    return _int(usage.get("cached_tokens"))


def _variant_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prompt_tokens = sum(_int(row.get("prompt_tokens")) for row in rows)
    cached_tokens = sum(_int(row.get("cached_tokens")) for row in rows)
    utility_success_count = sum(1 for row in rows if row.get("utility_ok") is True)
    return {
        "request_count": len(rows),
        "rewritten_count": sum(1 for row in rows if row.get("rewrite_applied") is True),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": sum(_int(row.get("completion_tokens")) for row in rows),
        "total_tokens": sum(_int(row.get("total_tokens")) for row in rows),
        "cached_tokens": cached_tokens,
        "cache_hit_ratio": cached_tokens / prompt_tokens if prompt_tokens else 0.0,
        "utility_success_count": utility_success_count,
        "utility_success_rate": utility_success_count / len(rows) if rows else 0.0,
        "avg_latency_seconds": (
            sum(float(row.get("latency_seconds") or 0.0) for row in rows) / len(rows) if rows else 0.0
        ),
    }


def _delta(baseline: Mapping[str, Any], plugin: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cached_tokens_delta": _int(plugin.get("cached_tokens")) - _int(baseline.get("cached_tokens")),
        "cache_hit_ratio_delta": float(plugin.get("cache_hit_ratio") or 0.0)
        - float(baseline.get("cache_hit_ratio") or 0.0),
        "utility_success_rate_delta": float(plugin.get("utility_success_rate") or 0.0)
        - float(baseline.get("utility_success_rate") or 0.0),
        "prompt_tokens_delta": _int(plugin.get("prompt_tokens")) - _int(baseline.get("prompt_tokens")),
    }


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay real AutoGen GroupChat captured requests against DeepSeek.")
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--run-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_replay_ab(
        capture_path=args.capture,
        output=args.output,
        max_records=args.max_records,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        run_id=args.run_id,
    )
    print(json.dumps({"variants": summary["variants"], "delta": summary["delta"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
