from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

from autogen_prefix_tree.utility_dataset_builder import (
    build_utility_validator_smoke_dataset,
    build_readiness_report,
    ensure_utility_dataset_layout,
    update_readiness_report_with_pytest,
)
from autogen_prefix_tree.utility_dataset_sources import (
    build_utility_smoke_tasks,
    inspect_humaneval_source,
    load_humaneval_smoke_tasks,
)
from autogen_prefix_tree.utility_oracles import (
    JsonSchemaOracle,
    PrivacyLeakOracle,
    RoleBoundaryOracle,
    StateConsistencyOracle,
    ToolTraceOracle,
    UnitTestOracle,
)


def test_utility_dataset_layout_can_be_created(tmp_path) -> None:
    ensure_utility_dataset_layout(tmp_path)

    assert (tmp_path / "datasets" / "sources" / "humaneval" / "raw").is_dir()
    assert (tmp_path / "datasets" / "utility_validator" / "tasks" / "smoke").is_dir()
    assert (tmp_path / "datasets" / "utility_validator" / "schema").is_dir()


def test_humaneval_loader_reads_existing_repo_source() -> None:
    report = inspect_humaneval_source(Path("."))
    assert report["readable"] is True
    assert report["contains_humaneval_tasks"] is True
    assert report["rows_with_prompt_test_entry_point"] > 0
    assert report["usable"] is True

    result = load_humaneval_smoke_tasks(Path("."), max_tasks=2, include_text=False, created_at="2026-06-12T00:00:00Z")
    assert len(result.tasks) == 2
    assert result.tasks[0]["dataset_source"] == "humaneval"
    assert result.tasks[0]["expected_oracle_type"] == "unit_test"
    assert "original_messages_ref" in result.tasks[0]
    assert "original_messages" not in result.tasks[0]
    assert result.tasks[0]["prompt_text_included"] is False


def test_humaneval_loader_skips_missing_source_with_prompt_safe_reason(tmp_path) -> None:
    result = load_humaneval_smoke_tasks(tmp_path, max_tasks=2, include_text=False, created_at="2026-06-12T00:00:00Z")

    assert result.tasks == ()
    assert result.source_reports["usable"] is False
    assert "HumanEval source missing" in result.source_reports["prompt_safe_reason"]


def test_synthetic_and_format_sources_generate_tasks() -> None:
    result = build_utility_smoke_tasks(
        repo_root=Path("."),
        source="synthetic_agent,format_protocol,role_privacy,history_state",
        max_tasks=20,
        include_text=False,
        created_at="2026-06-12T00:00:00Z",
    )
    sources = {task["dataset_source"] for task in result.tasks}

    assert {"synthetic_agent", "format_protocol", "role_privacy", "history_state"}.issubset(sources)
    assert any(task["expected_oracle_type"] == "json_schema" for task in result.tasks)
    assert any(task["expected_oracle_type"] == "tool_trace" for task in result.tasks)
    assert all(task["prompt_text_included"] is False for task in result.tasks)


def test_oracles_pass_and_fail_fake_outputs() -> None:
    assert UnitTestOracle({"entry_point": "foo"}).evaluate({}, {"tests_passed": True}).passed is True
    assert UnitTestOracle().evaluate({}, {"tests_passed": False}).failure_type == "code_test_fail"

    schema_oracle = JsonSchemaOracle(
        {
            "json_schema": {
                "type": "object",
                "required": ("status",),
                "properties": {"status": {"type": "string"}},
                "additionalProperties": False,
            }
        }
    )
    assert schema_oracle.evaluate({}, '{"status": "open"}').passed is True
    assert schema_oracle.evaluate({}, '{"status": "open", "debug": true}').failure_type == "json_schema_fail"

    tool_oracle = ToolTraceOracle({"required_tool_calls": ("lookup",), "forbidden_tool_calls": ("delete",)})
    assert tool_oracle.evaluate({}, {"tool_calls": [{"name": "lookup"}]}).passed is True
    assert tool_oracle.evaluate({}, {"tool_calls": [{"name": "delete"}]}).failure_type == "wrong_tool_call"

    assert PrivacyLeakOracle({"forbidden_substrings": ("SECRET",)}).evaluate({}, "public").passed is True
    assert PrivacyLeakOracle({"forbidden_substrings": ("SECRET",)}).evaluate({}, "SECRET").failure_type == "private_info_leak"

    assert RoleBoundaryOracle({"forbidden_role_claims": ("as the admin",)}).evaluate({}, "reviewer note").passed is True
    assert RoleBoundaryOracle({"forbidden_role_claims": ("as the admin",)}).evaluate({}, "as the admin").failure_type == "role_confusion"

    state_oracle = StateConsistencyOracle({"expected_event_order": ("a", "b")})
    assert state_oracle.evaluate({}, {"observed_event_order": ("a", "b")}).passed is True
    assert state_oracle.evaluate({}, {"observed_event_order": ("b", "a")}).failure_type == "state_mismatch"


def test_smoke_builder_generates_task_label_feature_without_prompt_text(tmp_path) -> None:
    result = build_utility_validator_smoke_dataset(
        repo_root=Path("."),
        output_root=tmp_path / "utility_validator",
        source="all",
        mode="smoke",
        backend="fake",
        max_tasks=10,
        include_text=False,
        created_at="2026-06-12T00:00:00Z",
    )

    tasks = _read_jsonl(Path(result.task_path))
    labels = _read_jsonl(Path(result.label_path))
    features = _read_jsonl(Path(result.feature_path))
    serialized = json.dumps({"tasks": tasks, "labels": labels, "features": features}, ensure_ascii=False)

    assert result.summary["task_count"] == 10
    assert result.summary["task_count"] == len(tasks) == len(labels) == len(features)
    assert result.summary["default_real_api_calls"] == 0
    assert result.summary["network_access_required"] is False
    assert result.summary["fake_smoke_label_count"] == len(labels)
    assert {task["dataset_source"] for task in tasks} >= {"humaneval", "synthetic_agent", "format_protocol"}
    assert all(task["prompt_text_included"] is False for task in tasks)
    assert all(label["label_source"] == "fake_smoke_oracle" for label in labels)
    assert all(feature["whether_label_is_utility_verified"] is False for feature in features)
    assert "def has_close_elements" not in serialized
    assert "SMOKE-PRIVATE-TOKEN-7" not in serialized


def test_smoke_builder_include_text_marks_artifacts(tmp_path) -> None:
    result = build_utility_validator_smoke_dataset(
        repo_root=Path("."),
        output_root=tmp_path / "utility_validator",
        source="format_protocol",
        mode="smoke",
        backend="fake",
        max_tasks=1,
        include_text=True,
        created_at="2026-06-12T00:00:00Z",
    )

    task = _read_jsonl(Path(result.task_path))[0]
    label = _read_jsonl(Path(result.label_path))[0]
    assert task["prompt_text_included"] is True
    assert label["prompt_text_included"] is True
    assert "original_prompt_text" in label["prompt_materialization"]


def test_dsapi_backend_requires_explicit_cost_confirmation(tmp_path) -> None:
    with pytest.raises(ValueError, match="confirm-cost-aware"):
        build_utility_validator_smoke_dataset(
            repo_root=Path("."),
            output_root=tmp_path / "utility_validator",
            source="format_protocol",
            backend="dsapi",
            max_tasks=1,
        )

    with pytest.raises(ValueError, match="smoke_real"):
        build_utility_validator_smoke_dataset(
            repo_root=Path("."),
            output_root=tmp_path / "utility_validator",
            source="format_protocol",
            backend="dsapi",
            max_tasks=1,
            max_api_calls=5,
            confirm_cost_aware=True,
        )


def test_dsapi_real_smoke_uses_fake_upstream_and_writes_cost_summary(tmp_path, monkeypatch) -> None:
    seen_requests: list[dict[str, Any]] = []
    server = _start_fake_dsapi(seen_requests)
    config_path = tmp_path / "dsapi_config.txt"
    config_path.write_text(
        "\n".join(
            [
                "api_key: sk-test-secret",
                "model: deepseek-v4-pro",
                f"base_url: http://127.0.0.1:{server.server_port}/v1",
            ]
        ),
        encoding="utf-8",
    )
    try:
        result = build_utility_validator_smoke_dataset(
            repo_root=Path("."),
            output_root=tmp_path / "utility_validator",
            source="format_protocol",
            mode="smoke_real",
            backend="dsapi",
            max_tasks=3,
            max_api_calls=5,
            confirm_cost_aware=True,
            model="v4flash",
            project_config_path=config_path,
            created_at="2026-06-12T00:00:00Z",
        )
    finally:
        server.shutdown()
        server.server_close()

    labels = _read_jsonl(Path(result.label_path))
    features = _read_jsonl(Path(result.feature_path))
    serialized = json.dumps({"summary": result.summary, "labels": labels, "features": features}, ensure_ascii=False)

    assert len(labels) == 3
    assert len(features) == 3
    assert all(label["label_source"] == "dsapi_execution_oracle" for label in labels)
    assert result.summary["mode"] == "smoke_real"
    assert result.summary["actual_real_api_calls"] <= 5
    assert result.summary["cost_summary"]["api_call_count"] == len(seen_requests)
    assert result.summary["cost_summary"]["total_input_tokens"] > 0
    assert result.summary["cost_summary"]["total_output_tokens"] > 0
    assert result.summary["cost_summary"]["total_cached_tokens"] > 0
    assert result.summary["cost_summary"]["estimated_cost_usd"] > 0
    assert result.summary["cost_summary"]["model"] == "deepseek-v4-flash"
    assert result.summary["cost_summary"]["model_is_v4_flash"] is True
    assert result.summary["cost_summary"]["model_is_pro"] is False
    assert result.summary["network_access_required"] is True
    assert result.summary["real_label_count"] == 3
    assert result.summary["fake_smoke_label_count"] == 0
    assert all(request["body"]["model"] == "deepseek-v4-flash" for request in seen_requests)
    assert all(label["input_tokens"] > 0 for label in labels)
    assert all(label["output_tokens"] > 0 for label in labels)
    assert all(label["cached_tokens"] > 0 for label in labels)
    assert all(label["latency"] > 0 for label in labels)
    assert all(label["estimated_cost"] > 0 for label in labels)
    assert "sk-test-secret" not in serialized
    assert all(_header_value(request["headers"], "authorization") == "Bearer sk-test-secret" for request in seen_requests)


def test_dsapi_real_smoke_refuses_over_budget_task_count(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-secret")
    with pytest.raises(ValueError, match="requires DeepSeek V4 Flash"):
        build_utility_validator_smoke_dataset(
            repo_root=Path("."),
            output_root=tmp_path / "utility_validator",
            source="format_protocol",
            mode="smoke_real",
            backend="dsapi",
            max_tasks=1,
            max_api_calls=5,
            confirm_cost_aware=True,
        )


def test_dsapi_smoke_real_refuses_over_new_smoke_budget(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-secret")
    with pytest.raises(ValueError, match="at most 10 tasks"):
        build_utility_validator_smoke_dataset(
            repo_root=Path("."),
            output_root=tmp_path / "utility_validator",
            source="format_protocol",
            mode="smoke_real",
            backend="dsapi",
            max_tasks=11,
            max_api_calls=20,
            confirm_cost_aware=True,
            model="v4flash",
        )


def test_pilot_real_expands_tasks_and_writes_readiness(tmp_path) -> None:
    seen_requests: list[dict[str, Any]] = []
    server = _start_fake_dsapi(seen_requests)
    config_path = tmp_path / "dsapi_config.txt"
    config_path.write_text(
        "\n".join(
            [
                "api_key: sk-test-secret",
                "model: deepseek-v4-pro",
                f"base_url: http://127.0.0.1:{server.server_port}/v1",
            ]
        ),
        encoding="utf-8",
    )
    try:
        smoke = build_utility_validator_smoke_dataset(
            repo_root=Path("."),
            output_root=tmp_path / "utility_validator",
            source="all",
            mode="smoke_real",
            backend="dsapi",
            max_tasks=10,
            max_api_calls=20,
            confirm_cost_aware=True,
            model="v4flash",
            project_config_path=config_path,
            created_at="2026-06-12T00:00:00Z",
        )
        pilot = build_utility_validator_smoke_dataset(
            repo_root=Path("."),
            output_root=tmp_path / "utility_validator",
            source="all",
            mode="pilot_real",
            backend="dsapi",
            max_tasks=30,
            max_api_calls=100,
            confirm_cost_aware=True,
            model="v4flash",
            project_config_path=config_path,
            created_at="2026-06-12T00:00:00Z",
        )
    finally:
        server.shutdown()
        server.server_close()

    smoke_labels = _read_jsonl(Path(smoke.label_path))
    pilot_labels = _read_jsonl(Path(pilot.label_path))
    pilot_features = _read_jsonl(Path(pilot.feature_path))
    readiness = update_readiness_report_with_pytest(
        output_root=tmp_path / "utility_validator",
        pytest_q_passed=True,
        pytest_report={"command": "pytest -q", "exit_code": 0},
    )

    assert len(pilot_labels) == 30
    assert len(pilot_features) == 30
    assert {label["dataset_source"] for label in smoke_labels} == {
        "humaneval",
        "synthetic_agent",
        "format_protocol",
        "role_privacy",
        "history_state",
    }
    assert {label["dataset_source"] for label in pilot_labels} == {
        "humaneval",
        "synthetic_agent",
        "format_protocol",
        "role_privacy",
        "history_state",
    }
    assert all(label["label_source"] == "dsapi_execution_oracle" for label in pilot_labels)
    assert all(label["prompt_text_included"] is False for label in pilot_labels)
    assert pilot.summary["mode"] == "pilot_real"
    assert pilot.summary["actual_real_api_calls"] <= 100
    assert readiness["ready_for_full_real_labeling"] is True
    assert readiness["checks"]["model_is_v4_flash"] is True
    assert readiness["checks"]["model_is_not_pro"] is True


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _header_value(headers: dict[str, Any], name: str) -> str | None:
    lowered = name.lower()
    for key, value in headers.items():
        if str(key).lower() == lowered:
            return str(value)
    return None


def _start_fake_dsapi(seen_requests: list[dict[str, Any]]) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or "0")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            seen_requests.append({"body": body, "headers": dict(self.headers)})
            system_content = body["messages"][0]["content"]
            user_content = body["messages"][1]["content"]
            if "tests_passed" in system_content:
                content = {"status": "passed", "tests_passed": True}
            elif "tool_calls" in system_content or "tool_calls" in user_content:
                contract = user_content if "ORACLE_OUTPUT_CONTRACT" in user_content else system_content
                start = contract.find("[")
                end = contract.find("]", start)
                calls = json.loads(contract[start : end + 1]) if start >= 0 and end > start else []
                final_marker = "Use this exact final_result: "
                final_result = "ok"
                if final_marker in contract:
                    raw = contract.split(final_marker, 1)[1].split(".", 1)[0]
                    final_result = json.loads(raw)
                elif '"final_result":' in contract:
                    raw = contract.split('"final_result":', 1)[1].split(".", 1)[0].strip()
                    final_result = json.loads(raw)
                content = {"status": "passed", "tool_calls": calls, "final_result": final_result}
            elif "leaked_private_memory" in system_content:
                content = {
                    "status": "passed",
                    "text": "Public summary only. No private memory is included.",
                    "leaked_private_memory": False,
                }
            elif "role_boundary_ok" in system_content:
                content = {"status": "passed", "text": "Role-safe result.", "role_boundary_ok": True}
            elif "observed_event_order" in system_content:
                marker = '"observed_event_order":'
                raw_order = system_content.split(marker, 1)[1].rsplit("}", 1)[0]
                content = {"status": "passed", "state_consistent": True, "observed_event_order": json.loads(raw_order)}
            elif "ticket id T-42" in user_content:
                content = {"ticket_id": "T-42", "status": "open"}
            elif "Answer yes" in user_content:
                content = {"answer": "yes", "confidence": 0.9}
            else:
                content = {"status": "open", "priority": "high"}
            prompt_tokens = 100 + len(seen_requests)
            completion_tokens = 12
            response = {
                "choices": [{"message": {"content": json.dumps(content)}}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": 8},
                },
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server
