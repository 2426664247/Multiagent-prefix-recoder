from __future__ import annotations

import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

from autogen_prefix_tree.dataset_eval import evaluate_dataset
from autogen_prefix_tree.openai_forward_proxy import build_forward_proxy_server
from autogen_prefix_tree.openai_forward_proxy import _extract_usage
from autogen_prefix_tree.openai_forward_proxy import _join_url
from tests.test_openai_request_adapter import _body


def test_openai_forward_proxy_rewrites_warm_request_before_upstream(tmp_path) -> None:
    upstream_requests: list[dict[str, Any]] = []
    upstream = _start_fake_upstream(upstream_requests)
    telemetry_path = tmp_path / "proxy_telemetry.jsonl"
    proxy = build_forward_proxy_server(
        host="127.0.0.1",
        port=0,
        upstream_base_url=f"http://127.0.0.1:{upstream.server_port}",
        session_id="forward-proxy-test",
        telemetry_log_path=telemetry_path,
    )
    proxy_thread = Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    try:
        first = _post_json(proxy.server_port, _body("planner", api_key="secret-value"))
        second = _post_json(proxy.server_port, _body("engineer", api_key="secret-value"))
    finally:
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()

    assert first["choices"][0]["message"]["content"] == "ok"
    assert second["choices"][0]["message"]["content"] == "ok"
    assert len(upstream_requests) == 2
    assert upstream_requests[0]["headers"]["Authorization"] == "Bearer test-token"
    assert upstream_requests[1]["body"]["messages"][0]["content"].startswith("USER_TASK_START")
    assert upstream_requests[1]["body"]["api_key"] == "secret-value"

    telemetry_rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    adapter_rows = [
        row for row in telemetry_rows if row["schema_version"] == "prefix-openai-request-adapter-telemetry-v1"
    ]
    provider_rows = [
        row for row in telemetry_rows if row["schema_version"] == "prefix-forward-proxy-provider-telemetry-v1"
    ]
    assert adapter_rows[0]["validation"]["reason"] == "no_rewrite_needed"
    assert adapter_rows[1]["validation"]["reason"] == "validated"
    assert provider_rows[0]["rewrite_applied"] is False
    assert provider_rows[1]["rewrite_applied"] is True
    assert provider_rows[1]["actual_prompt_tokens"] == 123
    assert provider_rows[1]["actual_cached_tokens"] == 64
    assert provider_rows[1]["actual_completion_tokens"] == 7
    assert provider_rows[1]["actual_total_tokens"] == 130
    assert provider_rows[1]["actual_cost_usd"] == 0.013
    assert provider_rows[1]["latency_seconds"] >= 0
    telemetry_text = json.dumps(telemetry_rows, ensure_ascii=False)
    assert "secret-value" not in telemetry_text
    assert "AGENT_NAME: engineer" not in telemetry_text

    provider_summary = evaluate_dataset(input_path=telemetry_path).summary["provider_trace"]
    assert provider_summary["supported"] is True
    assert provider_summary["record_count"] == 2
    assert provider_summary["transformed_count"] == 1
    assert provider_summary["actual_prompt_tokens"] == 246
    assert provider_summary["actual_cached_tokens"] == 128
    assert provider_summary["actual_completion_tokens"] == 14
    assert provider_summary["actual_total_tokens"] == 260
    assert provider_summary["actual_cost_usd"] == 0.026
    assert provider_summary["cost_supported"] is True


def test_openai_forward_proxy_invalid_json_returns_400(tmp_path) -> None:
    upstream = _start_fake_upstream([])
    proxy = build_forward_proxy_server(
        host="127.0.0.1",
        port=0,
        upstream_base_url=f"http://127.0.0.1:{upstream.server_port}",
        telemetry_log_path=tmp_path / "telemetry.jsonl",
    )
    proxy_thread = Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{proxy.server_port}/v1/chat/completions",
            data=b"{not-json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=10)
        except urllib.error.HTTPError as exc:
            status = exc.code
            body = json.loads(exc.read().decode("utf-8"))
        else:
            raise AssertionError("expected HTTPError")
    finally:
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()

    assert status == 400
    assert body["error"]["message"] == "Invalid JSON request body."


def test_openai_forward_proxy_disabled_records_provider_telemetry_without_rewrite(tmp_path) -> None:
    upstream_requests: list[dict[str, Any]] = []
    upstream = _start_fake_upstream(upstream_requests)
    telemetry_path = tmp_path / "disabled_proxy_telemetry.jsonl"
    proxy = build_forward_proxy_server(
        host="127.0.0.1",
        port=0,
        upstream_base_url=f"http://127.0.0.1:{upstream.server_port}",
        session_id="forward-proxy-disabled-test",
        telemetry_log_path=telemetry_path,
        enabled=False,
    )
    proxy_thread = Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    try:
        _post_json(proxy.server_port, _body("planner", api_key="secret-value"))
        _post_json(proxy.server_port, _body("engineer", api_key="secret-value"))
    finally:
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()

    assert len(upstream_requests) == 2
    assert upstream_requests[1]["body"]["messages"][0]["content"].startswith("ROLE_SPECIFIC_INSTRUCTION_START")
    telemetry_rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    adapter_rows = [
        row for row in telemetry_rows if row["schema_version"] == "prefix-openai-request-adapter-telemetry-v1"
    ]
    provider_rows = [
        row for row in telemetry_rows if row["schema_version"] == "prefix-forward-proxy-provider-telemetry-v1"
    ]
    assert [row["enabled"] for row in adapter_rows] == [False, False]
    assert [row["validation"]["reason"] for row in adapter_rows] == ["disabled", "disabled"]
    assert [row["rewrite_applied"] for row in provider_rows] == [False, False]
    assert evaluate_dataset(input_path=telemetry_path).summary["provider_trace"]["record_count"] == 2


def test_openai_forward_proxy_can_reset_existing_telemetry_at_startup(tmp_path) -> None:
    upstream_requests: list[dict[str, Any]] = []
    upstream = _start_fake_upstream(upstream_requests)
    telemetry_path = tmp_path / "proxy_telemetry.jsonl"
    telemetry_path.write_text('{"old": true}\n', encoding="utf-8")
    proxy = build_forward_proxy_server(
        host="127.0.0.1",
        port=0,
        upstream_base_url=f"http://127.0.0.1:{upstream.server_port}",
        session_id="forward-proxy-reset-test",
        telemetry_log_path=telemetry_path,
        reset_telemetry_log=True,
    )
    proxy_thread = Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    try:
        _post_json(proxy.server_port, _body("planner", api_key="secret-value"))
    finally:
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()

    telemetry_rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    assert telemetry_rows
    assert all(row.get("old") is not True for row in telemetry_rows)
    assert evaluate_dataset(input_path=telemetry_path).summary["provider_trace"]["record_count"] == 1


def test_openai_forward_proxy_can_override_upstream_auth_from_project_deepseek_config(tmp_path) -> None:
    upstream_requests: list[dict[str, Any]] = []
    upstream = _start_fake_upstream(upstream_requests)
    telemetry_path = tmp_path / "project_config_proxy_telemetry.jsonl"
    config_path = tmp_path / "config" / "config.txt"
    config_path.parent.mkdir()
    config_path.write_text("deepseek=ds-secret-value\ndeepseek-v4-pro\n", encoding="utf-8")
    proxy = build_forward_proxy_server(
        host="127.0.0.1",
        port=0,
        upstream_base_url=f"http://127.0.0.1:{upstream.server_port}",
        session_id="forward-proxy-project-config-test",
        telemetry_log_path=telemetry_path,
        use_project_deepseek_config=True,
        project_config_path=config_path,
    )
    proxy_thread = Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    try:
        _post_json(proxy.server_port, _body("planner", api_key="secret-value"))
    finally:
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()

    assert upstream_requests[0]["headers"]["Authorization"] == "Bearer ds-secret-value"
    telemetry_text = telemetry_path.read_text(encoding="utf-8")
    assert "ds-secret-value" not in telemetry_text


def test_openai_forward_proxy_records_shadow_trial_telemetry_without_changing_forwarding(tmp_path) -> None:
    upstream_requests: list[dict[str, Any]] = []
    upstream = _start_fake_upstream(upstream_requests)
    telemetry_path = tmp_path / "proxy_shadow_telemetry.jsonl"
    plan_path = tmp_path / "shadow_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "prefix-static-rule-shadow-trial-plan-v1",
                "trial_mode": "shadow_only",
                "trial_rules": [
                    {
                        "trial_rule_id": "shadow_rule:team-policy-none",
                        "feature_set": "semantic_hint+risk_tag_set",
                        "features": {"semantic_hint": "team_policy", "risk_tag_set": "none"},
                        "shadow_action": "prefer_review",
                        "support": 4,
                        "purity": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    proxy = build_forward_proxy_server(
        host="127.0.0.1",
        port=0,
        upstream_base_url=f"http://127.0.0.1:{upstream.server_port}",
        session_id="forward-proxy-shadow-test",
        telemetry_log_path=telemetry_path,
        shadow_trial_plan_path=plan_path,
    )
    proxy_thread = Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    try:
        _post_json(proxy.server_port, _body("planner", api_key="secret-value"))
        _post_json(proxy.server_port, _body("engineer", api_key="secret-value"))
    finally:
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()

    assert len(upstream_requests) == 2
    assert upstream_requests[1]["body"]["messages"][0]["content"].startswith("USER_TASK_START")
    telemetry_rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    adapter_rows = [
        row for row in telemetry_rows if row["schema_version"] == "prefix-openai-request-adapter-telemetry-v1"
    ]
    provider_rows = [
        row for row in telemetry_rows if row["schema_version"] == "prefix-forward-proxy-provider-telemetry-v1"
    ]
    assert adapter_rows[1]["shadow_trial"]["matched_rule_count"] == 1
    assert adapter_rows[1]["shadow_trial"]["matched_rules"][0]["trial_rule_id"] == "shadow_rule:team-policy-none"
    assert adapter_rows[1]["shadow_trial"]["validator_behavior_change_allowed"] is False
    assert provider_rows[1]["rewrite_applied"] is True


def test_openai_forward_proxy_semantic_guard_rejection_forwards_original_warm_request(tmp_path) -> None:
    upstream_requests: list[dict[str, Any]] = []
    judge_requests: list[dict[str, Any]] = []
    upstream = _start_fake_upstream(upstream_requests)
    judge = _start_fake_judge(
        judge_requests,
        {"passed": False, "reason": "private boundary uncertain", "checks": ["private_boundary"], "confidence": 0.62},
    )
    telemetry_path = tmp_path / "proxy_guard_telemetry.jsonl"
    proxy = build_forward_proxy_server(
        host="127.0.0.1",
        port=0,
        upstream_base_url=f"http://127.0.0.1:{upstream.server_port}",
        session_id="forward-proxy-guard-test",
        telemetry_log_path=telemetry_path,
        semantic_guard_base_url=f"http://127.0.0.1:{judge.server_port}/v1",
        semantic_guard_model="local-semantic-judge",
    )
    proxy_thread = Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    try:
        _post_json(proxy.server_port, _body("planner", api_key="secret-value"))
        _post_json(proxy.server_port, _body("engineer", api_key="secret-value"))
    finally:
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()
        judge.shutdown()
        judge.server_close()

    assert len(upstream_requests) == 2
    assert upstream_requests[1]["body"]["messages"][0]["content"].startswith("ROLE_SPECIFIC_INSTRUCTION_START")
    assert len(judge_requests) == 1
    telemetry_rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    adapter_rows = [
        row for row in telemetry_rows if row["schema_version"] == "prefix-openai-request-adapter-telemetry-v1"
    ]
    provider_rows = [
        row for row in telemetry_rows if row["schema_version"] == "prefix-forward-proxy-provider-telemetry-v1"
    ]
    assert adapter_rows[1]["validation"]["reason"] == "semantic_guard_failed:private_boundary_uncertain"
    assert adapter_rows[1]["semantic_guard"]["passed"] is False
    assert provider_rows[1]["rewrite_applied"] is False


def test_openai_forward_proxy_extracts_common_openai_compatible_usage_shapes() -> None:
    response_body = json.dumps(
        {
            "usage": {
                "input_tokens": 200,
                "output_tokens": 12,
                "input_tokens_details": {"cache_read": 80},
                "total_cost": "0.024",
            }
        }
    ).encode("utf-8")

    usage = _extract_usage(response_body)

    assert usage["prompt_tokens"] == 200
    assert usage["cached_tokens"] == 80
    assert usage["completion_tokens"] == 12
    assert usage["total_tokens"] == 212
    assert usage["cost_usd"] == 0.024


def test_openai_forward_proxy_join_url_does_not_duplicate_v1_prefix() -> None:
    assert _join_url("https://api.example/v1", "/v1/chat/completions") == "https://api.example/v1/chat/completions"
    assert _join_url("https://api.example", "/v1/chat/completions") == "https://api.example/v1/chat/completions"
    assert _join_url("https://api.example/v1/", "chat/completions") == "https://api.example/v1/chat/completions"


def _post_json(port: int, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer test-token"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _start_fake_upstream(seen_requests: list[dict[str, Any]]) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or "0")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            seen_requests.append(
                {
                    "path": self.path,
                    "headers": {str(key): str(value) for key, value in self.headers.items()},
                    "body": body,
                }
            )
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

        def log_message(self, format, *args) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _start_fake_judge(seen_requests: list[dict[str, Any]], judge_result: dict[str, Any]) -> ThreadingHTTPServer:
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

        def log_message(self, format, *args) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
