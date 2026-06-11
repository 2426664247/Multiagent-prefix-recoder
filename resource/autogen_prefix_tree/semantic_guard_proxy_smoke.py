from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Mapping, Sequence

from .dataset_eval import evaluate_dataset
from .openai_forward_proxy import build_forward_proxy_server


@dataclass(frozen=True)
class SemanticGuardProxySmokeResult:
    output_dir: str
    summary_path: str
    summary: dict[str, Any]


def run_semantic_guard_proxy_smoke(
    *,
    output_dir: str | Path,
    session_id: str = "semantic-guard-proxy-smoke",
    judge_mode: str = "reject",
    min_confidence: float = 0.75,
) -> SemanticGuardProxySmokeResult:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    telemetry_path = out / "semantic_guard_proxy_telemetry.jsonl"
    upstream_requests: list[dict[str, Any]] = []
    judge_requests: list[dict[str, Any]] = []
    upstream = _start_fake_upstream(upstream_requests)
    judge = _start_fake_judge(judge_requests, _judge_result(judge_mode))
    proxy = build_forward_proxy_server(
        host="127.0.0.1",
        port=0,
        upstream_base_url=f"http://127.0.0.1:{upstream.server_port}/v1",
        session_id=session_id,
        telemetry_log_path=telemetry_path,
        semantic_guard_base_url=f"http://127.0.0.1:{judge.server_port}/v1",
        semantic_guard_model="fake-local-semantic-judge",
        semantic_guard_min_confidence=min_confidence,
    )
    proxy_thread = Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    try:
        _post_json(proxy.server_port, _body("planner"))
        _post_json(proxy.server_port, _body("engineer"))
    finally:
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()
        judge.shutdown()
        judge.server_close()

    telemetry_rows = _read_jsonl(telemetry_path)
    adapter_rows = [row for row in telemetry_rows if row.get("schema_version") == "prefix-openai-request-adapter-telemetry-v1"]
    provider_rows = [
        row for row in telemetry_rows if row.get("schema_version") == "prefix-forward-proxy-provider-telemetry-v1"
    ]
    provider_summary = evaluate_dataset(input_path=telemetry_path).summary["provider_trace"]
    warm_adapter = adapter_rows[-1] if adapter_rows else {}
    warm_provider = provider_rows[-1] if provider_rows else {}
    semantic_guard = warm_adapter.get("semantic_guard") if isinstance(warm_adapter.get("semantic_guard"), Mapping) else {}
    summary = {
        "schema_version": "prefix-semantic-guard-proxy-smoke-summary-v1",
        "output_dir": str(out),
        "session_id": session_id,
        "judge_mode": judge_mode,
        "fake_upstream": True,
        "fake_local_judge": True,
        "real_provider_metrics_available": False,
        "real_semantic_model_metrics_available": False,
        "request_count": len(provider_rows),
        "judge_request_count": len(judge_requests),
        "upstream_request_count": len(upstream_requests),
        "adapter_validation_reasons": [
            str((row.get("validation") or {}).get("reason") or "unknown")
            for row in adapter_rows
            if isinstance(row.get("validation"), Mapping)
        ],
        "provider_rewrite_applied": [bool(row.get("rewrite_applied")) for row in provider_rows],
        "warm_semantic_guard": semantic_guard,
        "warm_validation": warm_adapter.get("validation") if isinstance(warm_adapter.get("validation"), Mapping) else {},
        "warm_provider_rewrite_applied": bool(warm_provider.get("rewrite_applied")),
        "provider_summary": provider_summary,
        "telemetry_path": str(telemetry_path),
        "prompt_safe_summary": True,
        "limits": (
            "This smoke uses a fake upstream and fake local semantic judge. It validates proxy wiring and "
            "fail-closed/allow telemetry only; it does not prove real provider or semantic model quality."
        ),
    }
    summary_path = out / "semantic_guard_proxy_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return SemanticGuardProxySmokeResult(output_dir=str(out), summary_path=str(summary_path), summary=summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a no-network OpenAI-compatible proxy smoke with a fake local semantic guard judge."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--session-id", default="semantic-guard-proxy-smoke")
    parser.add_argument(
        "--judge-mode",
        choices=("accept", "reject", "low-confidence"),
        default="reject",
        help="Fake judge response mode for the warm rewrite request.",
    )
    parser.add_argument("--min-confidence", type=float, default=0.75)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_semantic_guard_proxy_smoke(
        output_dir=args.output_dir,
        session_id=args.session_id,
        judge_mode=args.judge_mode,
        min_confidence=args.min_confidence,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


def _judge_result(mode: str) -> dict[str, Any]:
    if mode == "accept":
        return {
            "passed": True,
            "reason": "semantic invariants hold",
            "checks": ["agent_identity", "latest_instruction", "private_boundary"],
            "confidence": 0.93,
        }
    if mode == "low-confidence":
        return {
            "passed": True,
            "reason": "looks plausible",
            "checks": ["agent_identity", "latest_instruction"],
            "confidence": 0.42,
        }
    return {
        "passed": False,
        "reason": "private boundary uncertain",
        "checks": ["private_boundary"],
        "confidence": 0.62,
    }


def _post_json(port: int, body: Mapping[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer fake-smoke-token"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _body(agent: str) -> dict[str, Any]:
    return {
        "model": "unit-model",
        "messages": [
            {
                "role": "system",
                "content": "\n\n".join(
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
                ),
            }
        ],
        "tools": [{"type": "function", "function": {"name": "record_metric"}}],
        "tool_choice": "auto",
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }


def _start_fake_upstream(seen_requests: list[dict[str, Any]]) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or "0")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            seen_requests.append({"path": self.path, "body": body})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                        "usage": {
                            "prompt_tokens": 123,
                            "completion_tokens": 7,
                            "total_tokens": 130,
                            "prompt_tokens_details": {"cached_tokens": 64},
                            "cost_usd": 0.013,
                        },
                    }
                ).encode("utf-8")
            )

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _start_fake_judge(seen_requests: list[dict[str, Any]], judge_result: Mapping[str, Any]) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or "0")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            seen_requests.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(judge_result),
                                }
                            }
                        ]
                    }
                ).encode("utf-8")
            )

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if isinstance(value, dict):
                rows.append(value)
    return tuple(rows)


if __name__ == "__main__":
    raise SystemExit(main())
