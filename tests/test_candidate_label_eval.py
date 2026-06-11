from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import autogen_prefix_tree.candidate_label_eval as label_eval
from autogen_prefix_tree.candidate_label_eval import evaluate_candidate_labels


def test_candidate_label_eval_accepts_reviews_and_rejects_prompt_safe_rows(tmp_path) -> None:
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "labeled.jsonl"
    summary_path = tmp_path / "summary.json"
    rows = [
        _row("a", semantic_hint="tool_or_code_policy", confidence=0.72, risk_tags=[]),
        _row("b", semantic_hint="verification_policy", confidence=0.74, risk_tags=["conditional_instruction"]),
        _row("c", semantic_hint="team_policy", confidence=0.70, risk_tags=["agent_identity_boundary"]),
        _row("d", semantic_hint="long_stable_instruction", confidence=0.80, risk_tags=[]),
        _row("e", semantic_hint="procedure_policy", confidence=0.50, risk_tags=[]),
    ]
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = evaluate_candidate_labels(
        input_path=input_path,
        output_path=output_path,
        summary_path=summary_path,
    )

    assert result.summary["candidate_count"] == 5
    assert result.summary["label_counts"] == {"accept": 2, "reject": 1, "review": 2}
    assert result.summary["rule_label_counts"] == {"accept": 2, "reject": 1, "review": 2}
    assert result.summary["rule_to_final_label_counts"] == {
        "accept->accept": 2,
        "reject->reject": 1,
        "review->review": 2,
    }
    assert result.summary["static_safety_clamp_count"] == 0
    assert result.summary["review_queue_diagnostics"]["review_candidate_count"] == 2
    assert result.summary["review_queue_diagnostics"]["semantic_hint_counts"] == {
        "procedure_policy": 1,
        "team_policy": 1,
    }
    assert result.summary["review_queue_diagnostics"]["risk_tag_counts"] == {
        "agent_identity_boundary": 1,
    }
    assert result.summary["review_queue_diagnostics"]["local_judge_priority"] == (
        "high_risk_review_with_local_judge_and_static_clamps"
    )
    assert result.summary["accepted_semantic_hint_counts"] == {
        "tool_or_code_policy": 1,
        "verification_policy": 1,
    }
    labeled = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert labeled[2]["label"] == "review"
    assert labeled[2]["label_reason"] == "high_risk_boundary:agent_identity_boundary"
    assert labeled[2]["rule_label"] == "review"
    assert labeled[2]["rule_label_reason"] == "high_risk_boundary:agent_identity_boundary"
    assert labeled[3]["label"] == "reject"
    assert all("text" not in row for row in labeled)
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written_summary["label_counts"]["accept"] == 2
    assert written_summary["judge_mode"] == "rule"


def test_candidate_label_eval_openai_compatible_judge_requires_candidate_text(tmp_path) -> None:
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "labeled.jsonl"
    input_path.write_text(
        json.dumps(_row("a", semantic_hint="tool_or_code_policy", confidence=0.72, risk_tags=[])) + "\n",
        encoding="utf-8",
    )

    result = evaluate_candidate_labels(
        input_path=input_path,
        output_path=output_path,
        judge="openai-compatible",
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-test-model",
    )

    labeled = json.loads(output_path.read_text(encoding="utf-8"))
    assert labeled["label"] == "review"
    assert labeled["label_reason"] == "missing_candidate_text"
    assert labeled["rule_label"] == "accept"
    assert result.summary["judge_mode"] == "openai-compatible"
    assert result.summary["local_judge"]["model"] == "local-test-model"
    assert result.summary["review_queue_diagnostics"]["local_judge_priority"] == (
        "rerun_with_candidate_text_before_local_judge"
    )


def test_candidate_label_eval_openai_compatible_judge_clamps_high_risk_accept(
    tmp_path, monkeypatch
) -> None:
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "labeled.jsonl"
    row = _row(
        "a",
        semantic_hint="tool_or_code_policy",
        confidence=0.72,
        risk_tags=["agent_identity_boundary"],
    )
    row["text"] = "You are the coding agent. Use tools carefully."
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    def fake_call(config, candidate_row):
        assert config.model == "local-test-model"
        assert candidate_row["candidate_id"] == "a"
        return {"label": "accept", "reason": "stable tool policy", "confidence": 0.91}

    monkeypatch.setattr(label_eval, "_call_openai_compatible_judge", fake_call)

    evaluate_candidate_labels(
        input_path=input_path,
        output_path=output_path,
        judge="openai-compatible",
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-test-model",
    )

    labeled = json.loads(output_path.read_text(encoding="utf-8"))
    assert labeled["label"] == "review"
    assert labeled["model_label"] == "accept"
    assert labeled["label_reason"] == "model_accept_clamped_by_static_safety:agent_identity_boundary"


def test_candidate_label_eval_openai_compatible_judge_clamps_low_confidence_accept(
    tmp_path, monkeypatch
) -> None:
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "labeled.jsonl"
    row = _row("a", semantic_hint="verification_policy", confidence=0.72, risk_tags=[])
    row["text"] = "When you find an answer, verify it with evidence."
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    def fake_call(config, candidate_row):
        assert config.model == "local-test-model"
        assert candidate_row["candidate_id"] == "a"
        return {"label": "accept", "reason": "stable verification policy", "confidence": 0.51}

    monkeypatch.setattr(label_eval, "_call_openai_compatible_judge", fake_call)

    evaluate_candidate_labels(
        input_path=input_path,
        output_path=output_path,
        judge="openai-compatible",
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-test-model",
        min_confidence=0.66,
    )

    labeled = json.loads(output_path.read_text(encoding="utf-8"))
    assert labeled["label"] == "review"
    assert labeled["model_label"] == "accept"
    assert labeled["model_confidence"] == 0.51
    assert labeled["label_reason"] == "model_accept_clamped_low_confidence:0.51"


def test_candidate_label_eval_summarizes_rule_model_final_label_transitions(
    tmp_path, monkeypatch
) -> None:
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "labeled.jsonl"
    row = _row("a", semantic_hint="verification_policy", confidence=0.72, risk_tags=[])
    row["text"] = "When you find an answer, verify it with evidence."
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    def fake_call(config, candidate_row):
        return {"label": "accept", "reason": "stable verification policy", "confidence": 0.51}

    monkeypatch.setattr(label_eval, "_call_openai_compatible_judge", fake_call)

    result = evaluate_candidate_labels(
        input_path=input_path,
        output_path=output_path,
        judge="openai-compatible",
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-test-model",
        min_confidence=0.66,
    )

    assert result.summary["rule_label_counts"] == {"accept": 1}
    assert result.summary["model_label_counts"] == {"accept": 1}
    assert result.summary["rule_to_model_label_counts"] == {"accept->accept": 1}
    assert result.summary["model_to_final_label_counts"] == {"accept->review": 1}
    assert result.summary["rule_to_final_label_counts"] == {"accept->review": 1}
    assert result.summary["static_safety_clamp_count"] == 1


def test_candidate_label_eval_openai_compatible_judge_calls_local_http_server(tmp_path) -> None:
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "labeled.jsonl"
    summary_path = tmp_path / "summary.json"
    row = _row("a", semantic_hint="verification_policy", confidence=0.74, risk_tags=[])
    row["text"] = "When you find an answer, verify it with evidence."
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    seen_requests: list[dict] = []
    server = _start_fake_openai_compatible_server(seen_requests)
    try:
        result = evaluate_candidate_labels(
            input_path=input_path,
            output_path=output_path,
            summary_path=summary_path,
            judge="openai-compatible",
            local_judge_base_url=f"http://127.0.0.1:{server.server_port}/v1",
            local_judge_model="local-http-test-model",
        )
    finally:
        server.shutdown()
        server.server_close()

    labeled = json.loads(output_path.read_text(encoding="utf-8"))
    assert labeled["label"] == "accept"
    assert labeled["label_reason"] == "local_model:stable_verification_policy"
    assert labeled["model_label"] == "accept"
    assert labeled["model_confidence"] == 0.93
    assert result.summary["model_label_counts"] == {"accept": 1}
    assert result.summary["rule_label_counts"] == {"accept": 1}
    assert result.summary["rule_to_model_label_counts"] == {"accept->accept": 1}
    assert result.summary["model_to_final_label_counts"] == {"accept->accept": 1}
    assert result.summary["rule_to_final_label_counts"] == {"accept->accept": 1}
    assert json.loads(summary_path.read_text(encoding="utf-8"))["local_judge"]["model"] == "local-http-test-model"
    assert len(seen_requests) == 1
    assert seen_requests[0]["model"] == "local-http-test-model"
    assert seen_requests[0]["response_format"] == {"type": "json_object"}
    assert seen_requests[0]["messages"][1]["role"] == "user"


def test_candidate_label_eval_parses_structured_non_json_local_judge_output(tmp_path) -> None:
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "labeled.jsonl"
    row = _row("a", semantic_hint="verification_policy", confidence=0.74, risk_tags=[])
    row["text"] = "When you find an answer, verify it with evidence."
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers["Content-Length"])
            _ = json.loads(self.rfile.read(length).decode("utf-8"))
            payload = {
                "choices": [
                    {
                        "message": {
                            "content": "label: accept\nreason: stable verification policy\nconfidence: 93%"
                        }
                    }
                ]
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        evaluate_candidate_labels(
            input_path=input_path,
            output_path=output_path,
            judge="openai-compatible",
            local_judge_base_url=f"http://127.0.0.1:{server.server_port}/v1",
            local_judge_model="local-http-test-model",
        )
    finally:
        server.shutdown()
        server.server_close()

    labeled = json.loads(output_path.read_text(encoding="utf-8"))
    assert labeled["label"] == "accept"
    assert labeled["model_label"] == "accept"
    assert labeled["model_confidence"] == 0.93
    assert labeled["label_reason"] == "local_model:stable_verification_policy"


def test_candidate_label_eval_review_scope_only_calls_local_judge_for_rule_review_rows(
    tmp_path, monkeypatch
) -> None:
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "labeled.jsonl"
    rows = [
        _row("accepted", semantic_hint="verification_policy", confidence=0.74, risk_tags=[]),
        _row("reviewed", semantic_hint="procedure_policy", confidence=0.60, risk_tags=[]),
        _row("rejected", semantic_hint="long_stable_instruction", confidence=0.80, risk_tags=[]),
    ]
    for row in rows:
        row["text"] = f"candidate text for {row['candidate_id']}"
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    seen_candidate_ids: list[str] = []

    def fake_call(config, candidate_row):
        seen_candidate_ids.append(candidate_row["candidate_id"])
        return {"label": "review", "reason": "needs human check", "confidence": 0.82}

    monkeypatch.setattr(label_eval, "_call_openai_compatible_judge", fake_call)

    result = evaluate_candidate_labels(
        input_path=input_path,
        output_path=output_path,
        judge="openai-compatible",
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-test-model",
        local_judge_scope="review",
    )

    labeled = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    by_id = {row["candidate_id"]: row for row in labeled}
    assert seen_candidate_ids == ["reviewed"]
    assert by_id["accepted"]["label"] == "accept"
    assert by_id["accepted"]["local_judge_action"] == "skipped_by_scope"
    assert "model_label" not in by_id["accepted"]
    assert by_id["reviewed"]["label"] == "review"
    assert by_id["reviewed"]["local_judge_action"] == "model_called"
    assert by_id["reviewed"]["model_label"] == "review"
    assert by_id["rejected"]["label"] == "reject"
    assert by_id["rejected"]["local_judge_action"] == "skipped_by_scope"
    assert result.summary["local_judge_scope"] == "review"
    assert result.summary["local_judge_action_counts"] == {
        "model_called": 1,
        "skipped_by_scope": 2,
    }
    assert result.summary["local_judge_skipped_by_scope_label_counts"] == {
        "accept": 1,
        "reject": 1,
    }
    assert result.summary["model_label_counts"] == {"review": 1}
    assert result.summary["rule_to_model_label_counts"] == {"review->review": 1}
    assert result.summary["rule_to_final_label_counts"] == {
        "accept->accept": 1,
        "reject->reject": 1,
        "review->review": 1,
    }
    assert result.summary["local_judge_effectiveness_diagnostics"] == {
        "schema_version": "prefix-local-judge-effectiveness-diagnostics-v1",
        "rule_review_candidate_count": 1,
        "model_called_rule_review_count": 1,
        "resolved_rule_review_count": 0,
        "called_resolved_rule_review_count": 0,
        "remaining_rule_review_count": 1,
        "resolution_rate": 0.0,
        "called_resolution_rate": 0.0,
        "rule_review_final_label_counts": {"review": 1},
        "rule_review_model_label_counts": {"review": 1},
        "rule_review_local_judge_action_counts": {"model_called": 1},
    }


def test_candidate_label_eval_caps_local_judge_calls(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "labeled.jsonl"
    rows = [
        _row("first", semantic_hint="verification_policy", confidence=0.74, risk_tags=[]),
        _row("second", semantic_hint="tool_or_code_policy", confidence=0.72, risk_tags=[]),
        _row("third", semantic_hint="team_policy", confidence=0.70, risk_tags=[]),
    ]
    for row in rows:
        row["text"] = f"candidate text for {row['candidate_id']}"
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    seen_candidate_ids: list[str] = []

    def fake_call(config, candidate_row):
        seen_candidate_ids.append(candidate_row["candidate_id"])
        return {"label": "accept", "reason": "stable shared policy", "confidence": 0.92}

    monkeypatch.setattr(label_eval, "_call_openai_compatible_judge", fake_call)

    result = evaluate_candidate_labels(
        input_path=input_path,
        output_path=output_path,
        judge="openai-compatible",
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-test-model",
        max_local_judge_calls=1,
    )

    labeled = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    by_id = {row["candidate_id"]: row for row in labeled}
    assert seen_candidate_ids == ["third"]
    assert by_id["first"]["local_judge_action"] == "skipped_by_budget"
    assert by_id["second"]["label"] == "accept"
    assert by_id["second"]["local_judge_action"] == "skipped_by_budget"
    assert "model_label" not in by_id["second"]
    assert by_id["third"]["local_judge_action"] == "model_called"
    assert result.summary["max_local_judge_calls"] == 1
    assert result.summary["local_judge_budget_strategy"] == "priority_review_risk_confidence_chars_v1"
    assert result.summary["local_judge_action_counts"] == {
        "model_called": 1,
        "skipped_by_budget": 2,
    }
    assert result.summary["local_judge_skipped_by_budget_label_counts"] == {"accept": 2}
    assert result.summary["local_judge_budget_diagnostics"] == {
        "schema_version": "prefix-local-judge-budget-diagnostics-v1",
        "budget_limited": True,
        "max_local_judge_calls": 1,
        "budget_strategy": "priority_review_risk_confidence_chars_v1",
        "eligible_for_budget_count": 3,
        "model_called_count": 1,
        "skipped_by_budget_count": 2,
        "budget_exhausted": True,
        "budget_coverage_rate": 1 / 3,
    }
    assert result.summary["model_label_counts"] == {"accept": 1}
    assert result.summary["rule_to_model_label_counts"] == {"accept->accept": 1}
    assert result.summary["rule_to_final_label_counts"] == {"accept->accept": 3}
    assert result.summary["local_judge_effectiveness_diagnostics"]["rule_review_candidate_count"] == 0
    assert result.summary["local_judge_effectiveness_diagnostics"]["resolution_rate"] is None


def test_candidate_label_eval_budget_prioritizes_review_risk_over_input_order(
    tmp_path, monkeypatch
) -> None:
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "labeled.jsonl"
    rows = [
        _row("early_accept", semantic_hint="verification_policy", confidence=0.74, risk_tags=[]),
        _row("later_low_confidence", semantic_hint="procedure_policy", confidence=0.60, risk_tags=[]),
        _row(
            "later_high_risk",
            semantic_hint="team_policy",
            confidence=0.70,
            risk_tags=["agent_identity_boundary"],
        ),
    ]
    for row in rows:
        row["text"] = f"candidate text for {row['candidate_id']}"
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    seen_candidate_ids: list[str] = []

    def fake_call(config, candidate_row):
        seen_candidate_ids.append(candidate_row["candidate_id"])
        return {"label": "review", "reason": "needs semantic review", "confidence": 0.88}

    monkeypatch.setattr(label_eval, "_call_openai_compatible_judge", fake_call)

    result = evaluate_candidate_labels(
        input_path=input_path,
        output_path=output_path,
        judge="openai-compatible",
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-test-model",
        max_local_judge_calls=1,
    )

    labeled = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    by_id = {row["candidate_id"]: row for row in labeled}
    assert seen_candidate_ids == ["later_high_risk"]
    assert by_id["early_accept"]["local_judge_action"] == "skipped_by_budget"
    assert by_id["later_low_confidence"]["local_judge_action"] == "skipped_by_budget"
    assert by_id["later_high_risk"]["local_judge_action"] == "model_called"
    assert result.summary["local_judge_action_counts"] == {
        "model_called": 1,
        "skipped_by_budget": 2,
    }
    assert result.summary["local_judge_budget_strategy"] == "priority_review_risk_confidence_chars_v1"
    assert result.summary["local_judge_skipped_by_budget_label_counts"] == {
        "accept": 1,
        "review": 1,
    }
    assert result.summary["local_judge_budget_diagnostics"]["budget_exhausted"] is True
    assert result.summary["local_judge_budget_diagnostics"]["budget_coverage_rate"] == 1 / 3


def _row(candidate_id: str, *, semantic_hint: str, confidence: float, risk_tags: list[str]) -> dict:
    return {
        "schema_version": "prefix-semantic-candidate-v1",
        "candidate_id": candidate_id,
        "source_path": "source.py",
        "symbol": "SYSTEM_MESSAGE",
        "parent_hash": "parent",
        "text_hash": f"text-{candidate_id}",
        "semantic_hint": semantic_hint,
        "confidence": confidence,
        "char_count": 100,
        "line_count": 1,
        "risk_tags": risk_tags,
        "label": None,
        "label_source": None,
        "label_notes": None,
    }


def _start_fake_openai_compatible_server(seen_requests: list[dict]) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or "0")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            seen_requests.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "label": "accept",
                                    "reason": "stable verification policy",
                                    "confidence": "0.93",
                                }
                            )
                        }
                    }
                ]
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))

        def log_message(self, format, *args) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
