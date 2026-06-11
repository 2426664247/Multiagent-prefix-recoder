from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Mapping, Sequence

from .offline_semantic_matrix import run_offline_semantic_matrix


@dataclass(frozen=True)
class OfflineLocalJudgeMatrixSmokeResult:
    output_dir: str
    summary_path: str
    matrix_summary_path: str
    matrix_report_path: str
    summary: dict[str, Any]


def run_offline_local_judge_matrix_smoke(
    *,
    sources: Sequence[tuple[str, str | Path]],
    output_dir: str | Path,
    session_id: str = "offline-local-judge-matrix-smoke",
    include_tests: bool = False,
    min_prompt_chars: int = 40,
    min_confidence: float = 0.66,
    local_judge_scope: str = "review",
    max_local_judge_calls: int | None = 50,
    fake_judge_mode: str = "accept",
    local_judge_timeout: float = 30.0,
) -> OfflineLocalJudgeMatrixSmokeResult:
    out = Path(output_dir)
    matrix_dir = out / "matrix"
    out.mkdir(parents=True, exist_ok=True)

    judge_state: dict[str, Any] = {
        "request_count": 0,
        "semantic_hint_counts": Counter(),
        "risk_tag_counts": Counter(),
        "model_label_counts": Counter(),
    }
    judge_lock = Lock()
    judge = _start_fake_openai_compatible_judge(
        state=judge_state,
        lock=judge_lock,
        mode=fake_judge_mode,
    )
    try:
        matrix = run_offline_semantic_matrix(
            sources=sources,
            output_dir=matrix_dir,
            session_id=session_id,
            include_tests=include_tests,
            include_candidate_text=True,
            min_prompt_chars=min_prompt_chars,
            min_confidence=min_confidence,
            judge="openai-compatible",
            local_judge_base_url=f"http://127.0.0.1:{judge.server_port}/v1",
            local_judge_model="fake-local-matrix-judge",
            local_judge_timeout=local_judge_timeout,
            local_judge_scope=local_judge_scope,
            max_local_judge_calls=max_local_judge_calls,
        )
    finally:
        judge.shutdown()
        judge.server_close()

    aggregate = matrix.summary.get("aggregate") if isinstance(matrix.summary.get("aggregate"), Mapping) else {}
    budget = (
        aggregate.get("local_judge_budget_diagnostics")
        if isinstance(aggregate.get("local_judge_budget_diagnostics"), Mapping)
        else {}
    )
    review_queue = (
        aggregate.get("review_queue_diagnostics")
        if isinstance(aggregate.get("review_queue_diagnostics"), Mapping)
        else {}
    )
    with judge_lock:
        request_count = int(judge_state["request_count"])
        semantic_hint_counts = dict(sorted(judge_state["semantic_hint_counts"].items()))
        risk_tag_counts = dict(sorted(judge_state["risk_tag_counts"].items()))
        model_label_counts = dict(sorted(judge_state["model_label_counts"].items()))

    summary = {
        "schema_version": "prefix-offline-local-judge-matrix-smoke-summary-v1",
        "prompt_safe_summary": True,
        "output_dir": str(out),
        "session_id": session_id,
        "fake_local_judge": True,
        "fake_local_judge_mode": fake_judge_mode,
        "fake_local_judge_request_count": request_count,
        "fake_local_judge_semantic_hint_counts": semantic_hint_counts,
        "fake_local_judge_risk_tag_counts": risk_tag_counts,
        "fake_local_judge_model_label_counts": model_label_counts,
        "local_judge_scope": local_judge_scope,
        "max_local_judge_calls": max_local_judge_calls,
        "matrix_summary_path": matrix.summary_path,
        "matrix_report_path": matrix.report_path,
        "matrix_digest": {
            "source_count": matrix.summary.get("source_count"),
            "completed_source_count": aggregate.get("completed_source_count"),
            "supported_source_count": aggregate.get("supported_source_count"),
            "recommendation": aggregate.get("recommendation"),
            "local_judge_action_counts": aggregate.get("local_judge_action_counts"),
            "local_judge_budget_diagnostics": budget,
            "prompt_extraction_diagnostics": {
                "extraction_counts": aggregate.get("extraction_counts"),
            },
            "candidate_label_diagnostics": {
                "rule_label_counts": aggregate.get("rule_label_counts"),
                "model_label_counts": aggregate.get("model_label_counts"),
                "rule_to_model_label_counts": aggregate.get("rule_to_model_label_counts"),
                "model_to_final_label_counts": aggregate.get("model_to_final_label_counts"),
                "rule_to_final_label_counts": aggregate.get("rule_to_final_label_counts"),
                "static_safety_clamp_count": aggregate.get("static_safety_clamp_count"),
                "local_judge_effectiveness_diagnostics": aggregate.get(
                    "local_judge_effectiveness_diagnostics"
                ),
            },
            "review_queue_diagnostics": {
                "review_parent_block_count": review_queue.get("review_parent_block_count"),
                "review_source_file_count": review_queue.get("review_source_file_count"),
                "review_candidate_chars": review_queue.get("review_candidate_chars"),
                "risk_tag_counts": review_queue.get("risk_tag_counts"),
                "semantic_hint_counts": review_queue.get("semantic_hint_counts"),
                "label_reason_counts": review_queue.get("label_reason_counts"),
                "local_judge_priority_counts": review_queue.get("local_judge_priority_counts"),
            },
        },
        "real_provider_metrics_available": False,
        "real_provider_metrics_note": (
            "This smoke uses a fake local OpenAI-compatible judge and offline source prompts only. "
            "It validates local-judge matrix wiring and budget accounting, but cannot prove real semantic-model "
            "quality, provider cached tokens, latency, cost, or task success."
        ),
    }
    summary_path = out / "offline_local_judge_matrix_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return OfflineLocalJudgeMatrixSmokeResult(
        output_dir=str(out),
        summary_path=str(summary_path),
        matrix_summary_path=matrix.summary_path,
        matrix_report_path=matrix.report_path,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a no-network offline semantic matrix with a fake OpenAI-compatible local judge."
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source specification as label=path. May be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--session-id", default="offline-local-judge-matrix-smoke")
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument("--min-prompt-chars", type=int, default=40)
    parser.add_argument("--min-confidence", type=float, default=0.66)
    parser.add_argument(
        "--local-judge-scope",
        choices=("all", "review"),
        default="review",
        help="Call the fake local judge for all candidates or only rule-review candidates.",
    )
    parser.add_argument(
        "--max-local-judge-calls",
        type=int,
        default=50,
        help="Optional cap on actual fake local judge calls per matrix source/variant.",
    )
    parser.add_argument(
        "--fake-judge-mode",
        choices=("accept", "review", "reject", "low-confidence"),
        default="accept",
        help="Deterministic fake local judge response mode.",
    )
    parser.add_argument("--local-judge-timeout", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_offline_local_judge_matrix_smoke(
        sources=[_parse_source_spec(value) for value in args.source],
        output_dir=args.output_dir,
        session_id=args.session_id,
        include_tests=args.include_tests,
        min_prompt_chars=args.min_prompt_chars,
        min_confidence=args.min_confidence,
        local_judge_scope=args.local_judge_scope,
        max_local_judge_calls=args.max_local_judge_calls,
        fake_judge_mode=args.fake_judge_mode,
        local_judge_timeout=args.local_judge_timeout,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    completed = result.summary["matrix_digest"].get("completed_source_count")
    return 0 if _int(completed) > 0 else 1


def _start_fake_openai_compatible_judge(
    *,
    state: dict[str, Any],
    lock: Lock,
    mode: str,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or "0")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            candidate = _candidate_payload(body)
            judge_result = _fake_judge_result(mode, candidate)
            with lock:
                state["request_count"] += 1
                state["semantic_hint_counts"][str(candidate.get("semantic_hint") or "unknown")] += 1
                state["model_label_counts"][str(judge_result.get("label") or "missing")] += 1
                for tag in candidate.get("risk_tags") or ():
                    state["risk_tag_counts"][str(tag)] += 1
            response = {"choices": [{"message": {"content": json.dumps(judge_result)}}]}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))

        def log_message(self, format, *args) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _candidate_payload(request_body: Mapping[str, Any]) -> dict[str, Any]:
    messages = request_body.get("messages") if isinstance(request_body.get("messages"), list) else []
    user_message = next(
        (message for message in messages if isinstance(message, Mapping) and message.get("role") == "user"),
        {},
    )
    content = user_message.get("content") if isinstance(user_message, Mapping) else None
    if not isinstance(content, str):
        return {}
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _fake_judge_result(mode: str, candidate: Mapping[str, Any]) -> dict[str, Any]:
    if mode == "review":
        return {"label": "review", "reason": "fake_smoke_review", "confidence": 0.88}
    if mode == "reject":
        return {"label": "reject", "reason": "fake_smoke_reject", "confidence": 0.9}
    if mode == "low-confidence":
        return {"label": "accept", "reason": "fake_smoke_low_confidence_accept", "confidence": 0.42}
    return {"label": "accept", "reason": "fake_smoke_accept", "confidence": 0.93}


def _parse_source_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        path = Path(value)
        return path.name or "source", value
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"missing source label in {value!r}")
    if not path.strip():
        raise ValueError(f"missing source path in {value!r}")
    return label, path.strip()


def _int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
