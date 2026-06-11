from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from autogen_prefix_tree.offline_semantic_pipeline import run_offline_semantic_pipeline


def test_offline_semantic_pipeline_runs_all_stages(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "agent.py").write_text(
        '''
SYSTEM_MESSAGE = """
You are a helpful AI assistant.
When using code, you must indicate the script type in the code block.
Do not ask users to copy and paste the result.
When you find an answer, verify the answer carefully and include evidence.
"""
''',
        encoding="utf-8",
    )

    result = run_offline_semantic_pipeline(
        source_root=source_root,
        output_dir=tmp_path / "out",
        session_id="pipeline-test",
    )

    assert result.summary["source_prompt_corpus"]["prompt_count"] == 1
    assert result.summary["dataset_eval"]["supported_request_count"] == 1
    assert result.summary["candidate_label_eval"]["candidate_count"] >= 1
    assert result.summary["candidate_label_eval"]["review_queue_diagnostics"]["schema_version"] == (
        "prefix-review-queue-diagnostics-v1"
    )
    assert result.summary["candidate_utility_eval"]["candidate_count"] >= 1
    gap = result.summary["semantic_rule_gap_summary"]
    assert gap["schema_version"] == "prefix-semantic-rule-gap-summary-v1"
    assert gap["candidate_count"] >= 1
    assert gap["accepted_candidate_count"] >= 1
    assert gap["rule_gap_observed"] is True
    assert "offline evidence" in gap["interpretation"]
    assert result.summary["real_provider_metrics_available"] is False
    artifacts = result.summary["artifacts"]
    for path in artifacts.values():
        assert path
    assert result.summary["review_worklist"]["schema_version"] == "prefix-review-worklist-summary-v1"
    assert result.summary["review_worklist"]["include_text"] is False
    assert result.summary["review_worklist_with_text"] is None
    assert "review_worklist" in artifacts
    assert "review_worklist_with_text" not in artifacts
    written_summary = json.loads((tmp_path / "out" / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert written_summary["schema_version"] == "prefix-offline-semantic-pipeline-summary-v1"


def test_offline_semantic_pipeline_can_route_to_openai_compatible_judge(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "agent.py").write_text(
        '''
SYSTEM_MESSAGE = """
You are a helpful AI assistant.
When using code, you must indicate the script type in the code block.
"""
''',
        encoding="utf-8",
    )

    result = run_offline_semantic_pipeline(
        source_root=source_root,
        output_dir=tmp_path / "out",
        session_id="pipeline-local-judge-test",
        include_candidate_text=False,
        judge="openai-compatible",
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-test-model",
    )

    assert result.summary["candidate_label_eval"]["judge_mode"] == "openai-compatible"
    assert result.summary["candidate_label_eval"]["local_judge"]["model"] == "local-test-model"
    assert result.summary["candidate_label_eval"]["label_counts"]["review"] >= 1
    assert result.summary["semantic_rule_gap_summary"]["review_queue_observed"] is True
    assert result.summary["candidate_label_eval"]["review_queue_diagnostics"]["local_judge_priority"] == (
        "rerun_with_candidate_text_before_local_judge"
    )


def test_offline_semantic_pipeline_calls_openai_compatible_judge_end_to_end(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    prompt = '''
SYSTEM_MESSAGE = """
You are a helpful AI assistant.

When using code, you must indicate the script type in the code block.

When you find an answer, verify the answer carefully and include evidence.
"""
'''
    (source_root / "agent_a.py").write_text(prompt, encoding="utf-8")
    (source_root / "agent_b.py").write_text(prompt, encoding="utf-8")

    seen_requests: list[dict] = []
    server = _start_fake_candidate_judge(seen_requests)
    try:
        result = run_offline_semantic_pipeline(
            source_root=source_root,
            output_dir=tmp_path / "out",
            session_id="pipeline-local-http-judge",
            include_candidate_text=True,
            judge="openai-compatible",
            local_judge_base_url=f"http://127.0.0.1:{server.server_port}/v1",
            local_judge_model="local-http-test-model",
            min_confidence=0.66,
        )
    finally:
        server.shutdown()
        server.server_close()

    label_summary = result.summary["candidate_label_eval"]
    assert label_summary["judge_mode"] == "openai-compatible"
    assert label_summary["local_judge"]["model"] == "local-http-test-model"
    assert label_summary["rule_label_counts"] == {"accept": 4}
    assert label_summary["model_label_counts"] == {"accept": 4}
    assert label_summary["rule_to_model_label_counts"] == {"accept->accept": 4}
    assert label_summary["model_to_final_label_counts"] == {"accept->accept": 2, "accept->review": 2}
    assert label_summary["rule_to_final_label_counts"] == {"accept->accept": 2, "accept->review": 2}
    assert label_summary["static_safety_clamp_count"] == 2

    assert result.summary["candidate_utility_eval"]["eligible_candidate_count"] == 2
    assert result.summary["candidate_utility_eval"]["repeated_candidate_group_count"] == 1
    assert result.summary["candidate_utility_eval_with_review"]["eligible_candidate_count"] == 4
    assert result.summary["candidate_utility_eval_with_review"]["repeated_candidate_group_count"] == 2
    assert result.summary["review_worklist_with_text"]["include_text"] is True
    assert result.summary["review_worklist_with_text"]["local_judge_action_counts"] == {"model_called": 2}
    assert result.summary["artifacts"]["review_worklist_with_text"].endswith("review_worklist_with_text.jsonl")
    assert result.summary["semantic_rule_gap_summary"]["review_queue_observed"] is True
    assert len(seen_requests) == 4
    assert all(request["model"] == "local-http-test-model" for request in seen_requests)
    candidate_payload = json.loads(seen_requests[0]["messages"][1]["content"])
    assert "text" in candidate_payload
    assert "api_key" not in json.dumps(seen_requests, ensure_ascii=False).lower()


def test_offline_semantic_pipeline_review_scope_limits_local_judge_calls(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "agent.py").write_text(
        '''
SYSTEM_MESSAGE = """
When you find an answer, verify the answer carefully and include evidence from the available context.

Select the next step in the plan before responding with the final answer.
"""
''',
        encoding="utf-8",
    )

    seen_requests: list[dict] = []
    server = _start_fake_candidate_judge(seen_requests)
    try:
        result = run_offline_semantic_pipeline(
            source_root=source_root,
            output_dir=tmp_path / "out",
            session_id="pipeline-local-http-judge-review-scope",
            include_candidate_text=True,
            judge="openai-compatible",
            local_judge_base_url=f"http://127.0.0.1:{server.server_port}/v1",
            local_judge_model="local-http-test-model",
            local_judge_scope="review",
            min_confidence=0.66,
        )
    finally:
        server.shutdown()
        server.server_close()

    label_summary = result.summary["candidate_label_eval"]
    assert label_summary["local_judge_scope"] == "review"
    assert label_summary["local_judge_action_counts"] == {
        "model_called": 1,
        "skipped_by_scope": 1,
    }
    assert label_summary["local_judge_skipped_by_scope_label_counts"] == {"accept": 1}
    assert label_summary["rule_label_counts"] == {"accept": 1, "review": 1}
    assert label_summary["model_label_counts"] == {"accept": 1}
    assert label_summary["rule_to_model_label_counts"] == {"review->accept": 1}
    assert label_summary["rule_to_final_label_counts"] == {"accept->accept": 1, "review->accept": 1}
    assert label_summary["local_judge_effectiveness_diagnostics"] == {
        "schema_version": "prefix-local-judge-effectiveness-diagnostics-v1",
        "rule_review_candidate_count": 1,
        "model_called_rule_review_count": 1,
        "resolved_rule_review_count": 1,
        "called_resolved_rule_review_count": 1,
        "remaining_rule_review_count": 0,
        "resolution_rate": 1.0,
        "called_resolution_rate": 1.0,
        "rule_review_final_label_counts": {"accept": 1},
        "rule_review_model_label_counts": {"accept": 1},
        "rule_review_local_judge_action_counts": {"model_called": 1},
    }
    assert result.summary["review_worklist"]["local_judge_action_counts"] == {}
    assert len(seen_requests) == 1
    candidate_payload = json.loads(seen_requests[0]["messages"][1]["content"])
    assert candidate_payload["semantic_hint"] == "procedure_policy"


def test_offline_semantic_pipeline_caps_local_judge_calls(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    prompt = '''
SYSTEM_MESSAGE = """
When using code, you must indicate the script type in the code block.

When you find an answer, verify the answer carefully and include evidence.
"""
'''
    (source_root / "agent_a.py").write_text(prompt, encoding="utf-8")
    (source_root / "agent_b.py").write_text(prompt, encoding="utf-8")

    seen_requests: list[dict] = []
    server = _start_fake_candidate_judge(seen_requests)
    try:
        result = run_offline_semantic_pipeline(
            source_root=source_root,
            output_dir=tmp_path / "out",
            session_id="pipeline-local-http-judge-budget",
            include_candidate_text=True,
            judge="openai-compatible",
            local_judge_base_url=f"http://127.0.0.1:{server.server_port}/v1",
            local_judge_model="local-http-test-model",
            max_local_judge_calls=1,
        )
    finally:
        server.shutdown()
        server.server_close()

    label_summary = result.summary["candidate_label_eval"]
    assert label_summary["max_local_judge_calls"] == 1
    assert label_summary["local_judge_action_counts"] == {
        "model_called": 1,
        "skipped_by_budget": 3,
    }
    assert label_summary["local_judge_skipped_by_budget_label_counts"] == {"accept": 3}
    assert label_summary["model_label_counts"] == {"accept": 1}
    assert label_summary["rule_to_model_label_counts"] == {"accept->accept": 1}
    assert label_summary["rule_to_final_label_counts"] == {"accept->accept": 3, "accept->review": 1}
    assert result.summary["review_worklist"]["local_judge_action_counts"] == {"model_called": 1}
    assert len(seen_requests) == 1


def _start_fake_candidate_judge(seen_requests: list[dict]) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or "0")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            seen_requests.append(body)
            candidate = json.loads(body["messages"][1]["content"])
            if candidate["semantic_hint"] == "tool_or_code_policy":
                judge_result = {
                    "label": "accept",
                    "reason": "stable code policy but low confidence",
                    "confidence": 0.51,
                }
            else:
                judge_result = {
                    "label": "accept",
                    "reason": "stable verification policy",
                    "confidence": 0.93,
                }
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
