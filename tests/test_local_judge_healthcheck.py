from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

from autogen_prefix_tree.local_judge_healthcheck import main, run_local_judge_healthcheck


def test_local_judge_healthcheck_ready_with_valid_openai_compatible_response(tmp_path) -> None:
    seen_requests: list[dict[str, Any]] = []
    server = _start_fake_judge(
        seen_requests,
        {"label": "accept", "reason": "stable verification policy", "confidence": 0.92},
    )
    try:
        result = run_local_judge_healthcheck(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="local-healthcheck-model",
            api_key="secret-healthcheck-key",
            timeout=5,
            summary_path=tmp_path / "summary.json",
        )
    finally:
        server.shutdown()
        server.server_close()

    summary = result.summary
    serialized = json.dumps(summary, ensure_ascii=False)
    assert summary["schema_version"] == "prefix-local-judge-healthcheck-summary-v1"
    assert summary["prompt_safe_summary"] is True
    assert summary["request_sent"] is True
    assert summary["response_parse_ok"] is True
    assert summary["allowed_label"] is True
    assert summary["label"] == "accept"
    assert summary["confidence"] == 0.92
    assert summary["confidence_meets_min"] is True
    assert summary["synthetic_label_matches_expected"] is True
    assert summary["ready"] is True
    assert summary["recommendation"] == "run_small_local_judge_batch"
    assert summary["api_key_configured"] is True
    assert "secret-healthcheck-key" not in serialized
    assert "LOCAL_JUDGE_HEALTHCHECK_SYNTHETIC_TEXT" not in serialized
    assert "check evidence before giving the final answer" not in serialized
    assert len(seen_requests) == 1
    assert seen_requests[0]["headers"]["Authorization"] == "Bearer secret-healthcheck-key"
    assert seen_requests[0]["body"]["model"] == "local-healthcheck-model"
    user_payload = json.loads(seen_requests[0]["body"]["messages"][1]["content"])
    assert "LOCAL_JUDGE_HEALTHCHECK_SYNTHETIC_TEXT" in user_payload["text"]

    written = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert written == summary


def test_local_judge_healthcheck_rejects_invalid_label_without_prompt_text(tmp_path) -> None:
    seen_requests: list[dict[str, Any]] = []
    server = _start_fake_judge(
        seen_requests,
        {"label": "maybe", "reason": "not an allowed label", "confidence": 0.9},
    )
    try:
        result = run_local_judge_healthcheck(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="local-healthcheck-model",
            summary_path=tmp_path / "invalid.json",
        )
    finally:
        server.shutdown()
        server.server_close()

    summary = result.summary
    serialized = json.dumps(summary, ensure_ascii=False)
    assert summary["request_sent"] is True
    assert summary["response_parse_ok"] is True
    assert summary["allowed_label"] is False
    assert summary["label"] == "maybe"
    assert summary["ready"] is False
    assert summary["error_type"] == "ValueError"
    assert summary["error_reason"] == "invalid_model_label"
    assert summary["recommendation"] == "fix_local_judge_endpoint_or_json_schema"
    assert "not an allowed label" not in serialized
    assert "LOCAL_JUDGE_HEALTHCHECK_SYNTHETIC_TEXT" not in serialized


def test_local_judge_healthcheck_reports_non_json_response_safely() -> None:
    seen_requests: list[dict[str, Any]] = []
    server = _start_fake_judge(seen_requests, response_content="this is not json")
    try:
        result = run_local_judge_healthcheck(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="local-healthcheck-model",
        )
    finally:
        server.shutdown()
        server.server_close()

    summary = result.summary
    serialized = json.dumps(summary, ensure_ascii=False)
    assert summary["request_sent"] is True
    assert summary["response_parse_ok"] is False
    assert summary["ready"] is False
    assert summary["error_type"] == "JSONDecodeError"
    assert summary["error_reason"] == "response_content_not_json"
    assert "this is not json" not in serialized
    assert "LOCAL_JUDGE_HEALTHCHECK_SYNTHETIC_TEXT" not in serialized


def test_local_judge_healthcheck_cli_writes_summary_and_exit_codes(tmp_path) -> None:
    seen_requests: list[dict[str, Any]] = []
    server = _start_fake_judge(
        seen_requests,
        {"label": "review", "reason": "valid but not expected", "confidence": 0.82},
    )
    try:
        exit_code = main(
            [
                "--base-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--model",
                "local-healthcheck-model",
                "--summary",
                str(tmp_path / "cli-summary.json"),
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    assert exit_code == 0
    summary = json.loads((tmp_path / "cli-summary.json").read_text(encoding="utf-8"))
    assert summary["ready"] is True
    assert summary["allowed_label"] is True
    assert summary["label"] == "review"
    assert summary["synthetic_label_matches_expected"] is False
    assert summary["recommendation"] == "inspect_synthetic_label_before_large_batch"

    failed = main(
        [
            "--base-url",
            "",
            "--model",
            "local-healthcheck-model",
            "--summary",
            str(tmp_path / "failed-summary.json"),
        ]
    )
    assert failed == 1
    failed_summary = json.loads((tmp_path / "failed-summary.json").read_text(encoding="utf-8"))
    assert failed_summary["request_sent"] is False
    assert failed_summary["ready"] is False
    assert failed_summary["error_reason"] == "missing_base_url"


def _start_fake_judge(
    seen_requests: list[dict[str, Any]],
    judge_result: dict[str, Any] | None = None,
    *,
    response_content: str | None = None,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or "0")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            seen_requests.append(
                {
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": body,
                }
            )
            content = response_content if response_content is not None else json.dumps(judge_result)
            response = {"choices": [{"message": {"content": content}}]}
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
