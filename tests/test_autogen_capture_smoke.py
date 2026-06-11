from __future__ import annotations

import asyncio

from autogen_prefix_tree.autogen_capture_smoke import run_autogen_capture_smoke


def test_autogen_capture_smoke_marked_prompt_reaches_dataset_eval(tmp_path) -> None:
    result = asyncio.run(
        run_autogen_capture_smoke(
            capture_path=tmp_path / "marked_capture.jsonl",
            summary_path=tmp_path / "marked_summary.json",
            telemetry_path=tmp_path / "marked_telemetry.jsonl",
            session_id="marked-smoke",
            prompt_style="marked",
        )
    )

    assert result.message_count == 3
    assert result.summary["supported_request_count"] == 2
    assert result.summary["semantic_coverage_supported"] is True
    assert result.summary["validation_reason_counts"] == {"no_rewrite_needed": 1, "validated": 1}
    assert result.summary["moved_semantic_type_counts"] == {
        "global_task_background": 1,
        "shared_context": 1,
        "team_policy": 1,
    }


def test_autogen_capture_smoke_magentic_one_like_prompt_exposes_rule_gap(tmp_path) -> None:
    result = asyncio.run(
        run_autogen_capture_smoke(
            capture_path=tmp_path / "magentic_capture.jsonl",
            summary_path=tmp_path / "magentic_summary.json",
            telemetry_path=tmp_path / "magentic_telemetry.jsonl",
            session_id="magentic-smoke",
            prompt_style="magentic_one_like",
        )
    )

    assert result.summary["supported_request_count"] == 2
    assert result.summary["applied_count"] == 0
    assert result.summary["validation_reason_counts"] == {"no_rewrite_needed": 2}
    assert result.summary["reusable_prefix_request_count"] == 0
    assert result.summary["semantic_type_counts"]["role_identity"] == 2
    assert result.summary["semantic_type_counts"]["current_user_instruction"] == 3
    assert "global_task_background" not in result.summary["semantic_type_counts"]
    assert "shared_context" not in result.summary["semantic_type_counts"]
    assert "team_policy" not in result.summary["semantic_type_counts"]
    assert result.summary["movability_counts"]["order_sensitive"] >= 2
