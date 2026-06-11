from __future__ import annotations

import json

import pytest

from autogen_prefix_tree.dataset_eval import evaluate_dataset, openai_messages_to_autogen


def test_dataset_eval_summarizes_openai_compatible_messages_without_prompt_text(tmp_path) -> None:
    input_path = tmp_path / "requests.jsonl"
    telemetry_path = tmp_path / "telemetry.jsonl"
    summary_path = tmp_path / "summary.json"
    rows = [
        _chat_request("planner"),
        _chat_request("engineer"),
        _chat_request("reviewer"),
    ]
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = evaluate_dataset(
        input_path=input_path,
        telemetry_path=telemetry_path,
        summary_path=summary_path,
        session_id="realistic-trace",
    )

    summary = result.summary
    assert summary["input_record_count"] == 3
    assert summary["supported_request_count"] == 3
    assert summary["semantic_coverage_supported"] is True
    assert summary["validation_reason_counts"] == {"no_rewrite_needed": 1, "validated": 2}
    assert summary["applied_count"] == 2
    assert summary["fallback_count"] == 0
    assert summary["semantic_type_counts"]["global_task_background"] == 3
    assert summary["semantic_type_counts"]["shared_context"] == 3
    assert summary["semantic_type_counts"]["team_policy"] == 3
    assert summary["moved_semantic_type_counts"]["global_task_background"] == 2
    assert summary["moved_semantic_type_counts"]["shared_context"] == 2
    assert summary["total_estimated_gain_chars"] > 0
    assert summary["provider_trace"]["supported"] is False
    assert summary["rule_gap_diagnostics"]["repeated_nonprefix_system_block_count"] == 1

    telemetry_text = telemetry_path.read_text(encoding="utf-8")
    assert "AGENT_NAME: engineer" not in telemetry_text
    assert "Build the cache experiment." not in telemetry_text

    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written_summary["supported_request_count"] == 3


def test_dataset_eval_marks_hash_only_proxy_trace_as_not_semantic_coverage(tmp_path) -> None:
    input_path = tmp_path / "proxy_trace.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "request_id": "run:0001",
                        "session_id": "run",
                        "original_prompt_hash": "abc",
                        "forwarded_prompt_hash": "abc",
                        "actual_prompt_tokens": 100,
                        "actual_cached_tokens": 0,
                        "actual_completion_tokens": 10,
                        "actual_total_tokens": 110,
                        "actual_cost_usd": 0.11,
                        "estimated_cached_tokens": 0,
                        "local_input_tokens": 100,
                        "latency_seconds": 1.5,
                        "transformation_applied": False,
                        "error": None,
                    }
                ),
                json.dumps(
                    {
                        "request_id": "run:0002",
                        "session_id": "run",
                        "original_prompt_hash": "def",
                        "forwarded_prompt_hash": "ghi",
                        "actual_prompt_tokens": 120,
                        "actual_cached_tokens": 64,
                        "actual_completion_tokens": 20,
                        "actual_total_tokens": 140,
                        "actual_cost_usd": 0.14,
                        "estimated_cached_tokens": 64,
                        "local_input_tokens": 120,
                        "latency_seconds": 2.5,
                        "transformation_applied": True,
                        "error": None,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = evaluate_dataset(input_path=input_path)

    summary = result.summary
    assert summary["supported_request_count"] == 0
    assert summary["unsupported_record_count"] == 2
    assert summary["semantic_coverage_supported"] is False
    assert summary["skipped_reason_counts"] == {"missing_messages": 2}
    assert summary["provider_trace"]["supported"] is True
    assert summary["provider_trace"]["record_count"] == 2
    assert summary["provider_trace"]["actual_prompt_tokens"] == 220
    assert summary["provider_trace"]["actual_cached_tokens"] == 64
    assert summary["provider_trace"]["transformed_count"] == 1
    assert summary["provider_trace"]["cost_supported"] is True
    assert summary["provider_trace"]["actual_cost_usd"] == pytest.approx(0.25)
    assert summary["provider_trace"]["average_cost_usd"] == pytest.approx(0.125)
    assert summary["provider_trace"]["latency_supported"] is True
    assert summary["provider_trace"]["average_latency_seconds"] == pytest.approx(2.0)
    assert summary["provider_trace"]["p50_latency_seconds"] == pytest.approx(2.0)
    assert summary["provider_trace"]["p95_latency_seconds"] == pytest.approx(2.45)


def test_dataset_eval_summarizes_nested_provider_usage_shapes(tmp_path) -> None:
    input_path = tmp_path / "provider_usage.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "request_id": "provider-1",
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 10,
                            "prompt_tokens_details": {"cached_tokens": 40},
                            "cost": "0.11",
                        },
                        "latency_ms": 1200,
                        "rewrite_applied": False,
                        "error": None,
                    }
                ),
                json.dumps(
                    {
                        "request_id": "provider-2",
                        "usage": {
                            "input_tokens": "120",
                            "output_tokens": 20,
                            "input_tokens_details": {"cache_read": "64"},
                            "total_cost": 0.14,
                        },
                        "latency_seconds": 2.5,
                        "rewrite_applied": True,
                        "error": None,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = evaluate_dataset(input_path=input_path)

    provider = result.summary["provider_trace"]
    assert result.summary["semantic_coverage_supported"] is False
    assert provider["supported"] is True
    assert provider["record_count"] == 2
    assert provider["actual_prompt_tokens"] == 220
    assert provider["actual_cached_tokens"] == 104
    assert provider["actual_completion_tokens"] == 30
    assert provider["actual_total_tokens"] == 250
    assert provider["transformed_count"] == 1
    assert provider["actual_cost_usd"] == pytest.approx(0.25)
    assert provider["average_latency_seconds"] == pytest.approx(1.85)


def test_dataset_eval_summarizes_shadow_trial_telemetry_records(tmp_path) -> None:
    input_path = tmp_path / "shadow_telemetry.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": "prefix-openai-request-adapter-telemetry-v1",
                        "shadow_trial": {
                            "schema_version": "prefix-shadow-trial-telemetry-v1",
                            "matched_rule_count": 1,
                            "unsupported_rule_count": 0,
                            "trial_action_counts": {"prefer_review": 1},
                            "matched_rules": [
                                {
                                    "trial_rule_id": "shadow_rule:a",
                                    "shadow_action": "prefer_review",
                                    "review_status": "ready_for_shadow_trial",
                                }
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "schema_version": "prefix-openai-request-adapter-telemetry-v1",
                        "shadow_trial": {
                            "schema_version": "prefix-shadow-trial-telemetry-v1",
                            "matched_rule_count": 2,
                            "unsupported_rule_count": 1,
                            "trial_action_counts": {"force_reject": 2},
                            "matched_rules": [
                                {
                                    "trial_rule_id": "shadow_rule:a",
                                    "shadow_action": "force_reject",
                                    "review_status": "inspect_source_label_conflict",
                                },
                                {
                                    "trial_rule_id": "shadow_rule:b",
                                    "shadow_action": "force_reject",
                                    "review_status": "needs_more_source_diversity_before_promotion",
                                },
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = evaluate_dataset(input_path=input_path)

    shadow = result.summary["shadow_trial_trace"]
    assert shadow["supported"] is True
    assert shadow["record_count"] == 2
    assert shadow["record_with_match_count"] == 2
    assert shadow["matched_rule_observation_count"] == 3
    assert shadow["unique_matched_rule_count"] == 2
    assert shadow["trial_action_counts"] == {"force_reject": 2, "prefer_review": 1}
    assert shadow["unsupported_rule_observation_count"] == 1
    assert shadow["validator_behavior_change_allowed"] is False


def test_dataset_eval_reads_utf8_bom_jsonl(tmp_path) -> None:
    input_path = tmp_path / "bom_requests.jsonl"
    input_path.write_text(json.dumps(_chat_request("planner"), ensure_ascii=False) + "\n", encoding="utf-8-sig")

    result = evaluate_dataset(input_path=input_path)

    assert result.summary["input_record_count"] == 1
    assert result.summary["supported_request_count"] == 1


def test_dataset_eval_reports_repeated_natural_language_nonprefix_system_blocks(tmp_path) -> None:
    input_path = tmp_path / "magentic_like.jsonl"
    system_prompt = (
        "You are a helpful AI assistant.\n"
        "Solve tasks using your coding and language skills.\n"
        "When using code, you must indicate the script type in the code block.\n"
        "Do not ask users to copy and paste the result.\n"
        "When you find an answer, verify the answer carefully. Include verifiable evidence."
    )
    rows = [
        {"body": {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": "task 1"}]}},
        {"body": {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": "task 2"}]}},
    ]
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = evaluate_dataset(input_path=input_path, session_id="magentic-like")
    diagnostics = result.summary["rule_gap_diagnostics"]

    assert result.summary["applied_count"] == 0
    assert diagnostics["repeated_nonprefix_system_block_count"] == 1
    assert diagnostics["repeated_natural_language_hint_block_count"] == 1
    assert diagnostics["natural_language_hint_counts"]["tool_or_code_policy"] == 2
    assert diagnostics["natural_language_hint_counts"]["verification_policy"] == 2
    semantic_candidates = diagnostics["semantic_candidate_diagnostics"]
    assert semantic_candidates["candidate_count"] >= 1
    assert semantic_candidates["semantic_hint_counts"]["tool_or_code_policy"] >= 1
    assert semantic_candidates["semantic_hint_counts"]["verification_policy"] >= 1
    all_candidates = diagnostics["all_nonprefix_semantic_candidate_diagnostics"]
    assert all_candidates["candidate_count"] >= semantic_candidates["candidate_count"]
    assert all_candidates["parent_block_count"] >= semantic_candidates["parent_block_count"]
    assert diagnostics["top_repeated_nonprefix_system_blocks"][0]["count"] == 2
    assert "content_hash" in diagnostics["top_repeated_nonprefix_system_blocks"][0]
    assert system_prompt not in json.dumps(diagnostics, ensure_ascii=False)


def test_dataset_eval_can_opt_into_natural_language_segmentation(tmp_path) -> None:
    input_path = tmp_path / "natural_language.jsonl"
    system_prompt = (
        "You are a helpful AI assistant.\n"
        "When using code, indicate the script type in the code block.\n"
        "Verify the answer carefully and include evidence.\n"
        "Reply with concise final results."
    )
    rows = [
        {"body": {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": "task 1"}]}},
        {"body": {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": "task 2"}]}},
    ]
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    default_result = evaluate_dataset(input_path=input_path, session_id="natural-default")
    segmented_result = evaluate_dataset(
        input_path=input_path,
        session_id="natural-segmented",
        enable_natural_language_segmentation=True,
    )

    assert default_result.summary["applied_count"] == 0
    assert segmented_result.summary["natural_language_segmentation_enabled"] is True
    assert segmented_result.summary["applied_count"] == 0
    assert segmented_result.summary["fallback_count"] == 1
    assert segmented_result.summary["validation_reason_counts"] == {
        "calibrated_utility_risk:natural_language_segment": 1,
        "no_rewrite_needed": 1,
    }
    assert segmented_result.summary["moved_semantic_type_counts"]["team_policy"] >= 1
    assert segmented_result.summary["semantic_type_counts"]["shared_tool_description"] == 2
    assert "helpful AI assistant" not in json.dumps(segmented_result.summary, ensure_ascii=False)


def test_openai_message_converter_accepts_nested_content_parts() -> None:
    messages = openai_messages_to_autogen(
        [
            {"role": "system", "content": [{"type": "text", "text": "TEAM_POLICY_START\nPolicy\nTEAM_POLICY_END"}]},
            {"role": "user", "content": [{"type": "text", "text": "latest request"}], "name": "user"},
            {"role": "assistant", "content": None, "name": "assistant"},
            {"role": "tool", "content": {"result": "ok"}, "tool_call_id": "call-1", "name": "lookup"},
        ]
    )

    assert [message.type for message in messages] == [
        "SystemMessage",
        "UserMessage",
        "AssistantMessage",
        "FunctionExecutionResultMessage",
    ]


def _chat_request(agent: str) -> dict:
    return {
        "id": f"request-{agent}",
        "body": {
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
                            "SHARED_GROUPCHAT_CONTEXT_START\n"
                            "The team is comparing baseline and plugin prompt layouts.\n"
                            "SHARED_GROUPCHAT_CONTEXT_END",
                            "TEAM_POLICY_START\nDo not move private memory or latest user instructions.\nTEAM_POLICY_END",
                            "TOOL_SCHEMA_START\nrecord_metric(name: string, value: number) -> string\nTOOL_SCHEMA_END",
                            "CURRENT_TURN_INSTRUCTION_START\nReport status for this agent.\nCURRENT_TURN_INSTRUCTION_END",
                        ]
                    ),
                }
            ],
            "tools": [{"type": "function", "function": {"name": "record_metric"}}],
            "temperature": 0,
        },
    }
