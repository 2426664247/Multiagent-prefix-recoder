from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

from autogen_prefix_tree import OpenAICompatibleRequestAdapter, rewrite_openai_request_body
from autogen_prefix_tree.openai_request_adapter import main


def test_openai_request_adapter_warm_rewrites_openai_compatible_body_without_prompt_telemetry(tmp_path) -> None:
    telemetry_path = tmp_path / "adapter_telemetry.jsonl"
    adapter = OpenAICompatibleRequestAdapter(session_id="adapter-test", telemetry_log_path=telemetry_path)

    cold = adapter.rewrite_request_body(_body("planner", api_key="secret-value"))
    warm = adapter.rewrite_request_body(_body("engineer", api_key="secret-value"))

    assert cold.applied is False
    assert cold.rewritten_body["messages"][0]["content"] == cold.original_body["messages"][0]["content"]
    assert warm.applied is True
    rewritten_content = warm.rewritten_body["messages"][0]["content"]
    assert rewritten_content.startswith("USER_TASK_START")
    assert "ROLE_SPECIFIC_INSTRUCTION_START" in rewritten_content
    assert warm.rewritten_body["api_key"] == "secret-value"
    assert warm.rewritten_body["temperature"] == 0

    rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["validation"]["reason"] == "no_rewrite_needed"
    assert rows[1]["validation"]["reason"] == "validated"
    telemetry_text = json.dumps(rows, ensure_ascii=False)
    assert "secret-value" not in telemetry_text
    assert "AGENT_NAME: engineer" not in telemetry_text
    assert "Shared benchmark context." not in telemetry_text
    assert rows[1]["body_shape"]["keys"] == [
        "messages",
        "model",
        "response_format",
        "temperature",
        "tool_choice",
        "tools",
    ]


def test_rewrite_openai_request_body_one_shot_keeps_cold_request_unchanged() -> None:
    result = rewrite_openai_request_body(_body("planner"), session_id="one-shot")

    assert result.applied is False
    assert result.telemetry["validation"]["reason"] == "no_rewrite_needed"
    assert result.rewritten_body["messages"][0]["content"] == result.original_body["messages"][0]["content"]


def test_openai_request_adapter_cli_rewrites_jsonl(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    telemetry_path = tmp_path / "telemetry.jsonl"
    input_path.write_text(
        json.dumps(_body("planner"), ensure_ascii=False)
        + "\n"
        + json.dumps(_body("engineer"), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--telemetry",
            str(telemetry_path),
            "--session-id",
            "adapter-cli",
        ]
    )

    assert exit_code == 0
    rewritten = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rewritten[0]["messages"][0]["content"].startswith("ROLE_SPECIFIC_INSTRUCTION_START")
    assert rewritten[1]["messages"][0]["content"].startswith("USER_TASK_START")
    telemetry_rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    assert telemetry_rows[1]["validation"]["applied"] is True


def test_openai_request_adapter_cli_can_enable_natural_language_segmentation(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    telemetry_path = tmp_path / "telemetry.jsonl"
    input_path.write_text(
        json.dumps(_natural_language_body(), ensure_ascii=False)
        + "\n"
        + json.dumps(_natural_language_body(), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--telemetry",
            str(telemetry_path),
            "--session-id",
            "adapter-cli-nl",
            "--enable-natural-language-segmentation",
        ]
    )

    assert exit_code == 0
    rewritten = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rewritten[0]["messages"][0]["content"].startswith("You are")
    assert rewritten[1]["messages"][0]["content"].startswith("You are")
    assert rewritten[1]["messages"][0]["content"] == _natural_language_body()["messages"][0]["content"]
    telemetry_rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    assert telemetry_rows[1]["natural_language_segmentation_enabled"] is True
    assert telemetry_rows[1]["validation"]["applied"] is False
    assert telemetry_rows[1]["validation"]["fallback"] is True
    assert telemetry_rows[1]["validation"]["reason"] == "calibrated_utility_risk:natural_language_segment"
    assert telemetry_rows[1]["semantic_type_counts"]["shared_tool_description"] == 1
    assert len(telemetry_rows[1]["cacheable_prefix_blocks"]) == 2


def test_openai_request_adapter_can_enable_groupchat_history_reordering(tmp_path) -> None:
    telemetry_path = tmp_path / "history_telemetry.jsonl"
    adapter = OpenAICompatibleRequestAdapter(
        session_id="adapter-history",
        telemetry_log_path=telemetry_path,
        enable_groupchat_history_reordering=True,
    )
    task = {"role": "user", "content": "Solve the task.", "name": "user"}
    planner_history = {"role": "user", "content": "Planner: write tests first.", "name": "planner"}
    engineer_history = {"role": "user", "content": "Engineer: implementation is ready.", "name": "engineer"}

    adapter.rewrite_request_body(_groupchat_body("planner", task))
    adapter.rewrite_request_body(_groupchat_body("engineer", task, planner_history))
    warm = adapter.rewrite_request_body(_groupchat_body("reviewer", task, planner_history, engineer_history))

    assert warm.applied is True
    messages = warm.rewritten_body["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("USER_TASK_START")
    assert messages[1]["name"] == "user"
    assert messages[2]["name"] == "planner"
    assert messages[3]["role"] == "system"
    assert "AGENT_NAME: reviewer" in messages[3]["content"]
    telemetry_rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    assert telemetry_rows[-1]["groupchat_history_reordering_enabled"] is True
    assert telemetry_rows[-1]["validation"]["reason"] == "validated"
    assert telemetry_rows[-1]["semantic_type_counts"]["conversation_history"] == 2


def test_openai_request_adapter_shadow_trial_records_metadata_matches_without_changing_body(tmp_path) -> None:
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
                    },
                    {
                        "trial_rule_id": "shadow_rule:unsupported",
                        "feature_set": "rule_label+semantic_hint+risk_tag_set",
                        "features": {
                            "rule_label": "review",
                            "semantic_hint": "team_policy",
                            "risk_tag_set": "none",
                        },
                        "shadow_action": "force_reject",
                        "support": 2,
                        "purity": 1.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    plain = OpenAICompatibleRequestAdapter(session_id="shadow-plain")
    shadow = OpenAICompatibleRequestAdapter(
        session_id="shadow-enabled",
        shadow_trial_plan_path=plan_path,
    )

    plain.rewrite_request_body(_body("planner"))
    plain_warm = plain.rewrite_request_body(_body("engineer"))
    shadow.rewrite_request_body(_body("planner"))
    shadow_warm = shadow.rewrite_request_body(_body("engineer"))

    assert shadow_warm.rewritten_body == plain_warm.rewritten_body
    shadow_trial = shadow_warm.telemetry["shadow_trial"]
    assert shadow_trial["schema_version"] == "prefix-shadow-trial-telemetry-v1"
    assert shadow_trial["prompt_safe_summary"] is True
    assert shadow_trial["trial_mode"] == "shadow_only"
    assert shadow_trial["trial_rule_count"] == 2
    assert shadow_trial["matched_rule_count"] == 1
    assert shadow_trial["matched_rules"][0]["trial_rule_id"] == "shadow_rule:team-policy-none"
    assert shadow_trial["matched_rules"][0]["shadow_action"] == "prefer_review"
    assert shadow_trial["unsupported_rule_count"] == 1
    assert shadow_trial["unsupported_rules"][0]["unsupported_feature_names"] == ["rule_label"]
    assert shadow_trial["validator_behavior_change_allowed"] is False
    assert shadow_trial["safe_for_automatic_validator_promotion"] is False
    serialized = json.dumps(shadow_trial, ensure_ascii=False)
    assert "Shared benchmark context." not in serialized
    assert "AGENT_NAME: engineer" not in serialized


def test_openai_request_adapter_cli_semantic_guard_rejection_keeps_warm_body_original(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    telemetry_path = tmp_path / "telemetry.jsonl"
    input_path.write_text(
        json.dumps(_body("planner"), ensure_ascii=False)
        + "\n"
        + json.dumps(_body("engineer"), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    seen_requests: list[dict[str, Any]] = []
    server = _start_fake_judge(
        seen_requests,
        {"passed": False, "reason": "latest instruction uncertain", "checks": ["latest_instruction"], "confidence": 0.67},
    )
    try:
        exit_code = main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--telemetry",
                str(telemetry_path),
                "--session-id",
                "adapter-cli-guard",
                "--semantic-guard-base-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--semantic-guard-model",
                "local-semantic-judge",
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    assert exit_code == 0
    rewritten = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rewritten[1]["messages"][0]["content"].startswith("ROLE_SPECIFIC_INSTRUCTION_START")
    telemetry_rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    assert telemetry_rows[1]["validation"]["fallback"] is True
    assert telemetry_rows[1]["validation"]["reason"] == "semantic_guard_failed:latest_instruction_uncertain"
    assert telemetry_rows[1]["semantic_guard"]["passed"] is False
    assert len(seen_requests) == 1


def test_openai_request_adapter_invalid_messages_fallbacks_without_mutating_body() -> None:
    adapter = OpenAICompatibleRequestAdapter(session_id="invalid")
    body = {"model": "unit-model", "messages": "not-a-list"}

    result = adapter.rewrite_request_body(body)

    assert result.fallback is True
    assert result.telemetry["validation"]["reason"] == "pipeline_exception:ValueError"
    assert result.rewritten_body == body


def _body(agent: str, *, api_key: str | None = None) -> dict:
    body = {
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
    if api_key is not None:
        body["api_key"] = api_key
    return body


def _natural_language_body() -> dict:
    return {
        "model": "unit-model",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant.\n"
                    "When using code, indicate the script type in the code block.\n"
                    "Verify the answer carefully and include evidence.\n"
                    "Reply with concise final results."
                ),
            }
        ],
        "temperature": 0,
    }


def _groupchat_body(agent: str, *messages: dict[str, Any]) -> dict:
    return {
        "model": "unit-model",
        "messages": [
            {
                "role": "system",
                "content": _body(agent)["messages"][0]["content"],
            },
            *messages,
        ],
        "temperature": 0,
    }


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
