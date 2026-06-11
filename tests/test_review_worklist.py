from __future__ import annotations

import json

from autogen_prefix_tree.review_worklist import build_review_worklist, main


def test_review_worklist_exports_prompt_safe_prioritized_rows_by_default(tmp_path) -> None:
    input_path = tmp_path / "labeled.jsonl"
    output_path = tmp_path / "review_worklist.jsonl"
    summary_path = tmp_path / "review_worklist_summary.json"
    rows = [
        _row("accept", label="accept", reason="accepted_by_conservative_rule_judge", risk_tags=[]),
        _row(
            "low",
            label="review",
            reason="low_confidence",
            risk_tags=[],
            text="low confidence prompt text",
            local_judge_action="model_called",
        ),
        _row(
            "risk",
            label="review",
            reason="high_risk_boundary:agent_identity_boundary",
            risk_tags=["agent_identity_boundary"],
            text="sensitive identity prompt text",
            char_count=200,
            local_judge_action="skipped_by_scope",
        ),
    ]
    _write_jsonl(input_path, rows)

    result = build_review_worklist(
        input_path=input_path,
        output_path=output_path,
        summary_path=summary_path,
    )

    exported = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["candidate_id"] for row in exported] == ["risk", "low"]
    assert exported[0]["review_priority"] == "high"
    assert exported[0]["recommended_action"] == "rerun_worklist_with_include_text_for_local_judge"
    assert exported[0]["local_judge_action"] == "skipped_by_scope"
    assert exported[1]["recommended_action"] == "human_review_local_judge_result"
    assert "text" not in exported[0]
    serialized = json.dumps(exported, ensure_ascii=False)
    assert "sensitive identity prompt text" not in serialized
    assert result.summary["review_candidate_count"] == 2
    assert result.summary["priority_counts"] == {"high": 1, "medium": 1}
    assert result.summary["local_judge_action_counts"] == {"model_called": 1, "skipped_by_scope": 1}
    assert result.summary["prompt_safe_summary"] is True
    assert result.summary["text_artifact_note"] == "output JSONL is prompt-safe metadata only"
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["schema_version"] == "prefix-review-worklist-summary-v1"


def test_review_worklist_can_explicitly_include_text_for_local_judge(tmp_path) -> None:
    input_path = tmp_path / "labeled.jsonl"
    output_path = tmp_path / "review_worklist_with_text.jsonl"
    _write_jsonl(
        input_path,
        [
            _row(
                "risk",
                label="review",
                reason="high_risk_boundary:agent_identity_boundary",
                risk_tags=["agent_identity_boundary"],
                text="candidate text for local judge",
                local_judge_action="model_called",
            )
        ],
    )

    result = build_review_worklist(input_path=input_path, output_path=output_path, include_text=True)

    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported["text"] == "candidate text for local judge"
    assert exported["local_judge_action"] == "model_called"
    assert exported["recommended_action"] == "human_review_local_judge_result"
    assert result.summary["include_text"] is True
    assert result.summary["local_judge_action_counts"] == {"model_called": 1}
    assert "raw candidate text" in result.summary["text_artifact_note"]


def test_review_worklist_flags_model_accept_static_clamps(tmp_path) -> None:
    input_path = tmp_path / "labeled.jsonl"
    output_path = tmp_path / "review_worklist.jsonl"
    _write_jsonl(
        input_path,
        [
            _row(
                "clamped",
                label="review",
                reason="model_accept_clamped_by_static_safety:agent_identity_boundary",
                risk_tags=["agent_identity_boundary"],
                text="candidate text for local judge",
                local_judge_action="model_called",
            )
        ],
    )

    result = build_review_worklist(input_path=input_path, output_path=output_path, include_text=True)

    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported["recommended_action"] == "review_model_accept_static_clamp"
    assert result.summary["recommended_action_counts"] == {"review_model_accept_static_clamp": 1}


def test_review_worklist_flags_budget_skipped_rows(tmp_path) -> None:
    input_path = tmp_path / "labeled.jsonl"
    output_path = tmp_path / "review_worklist.jsonl"
    _write_jsonl(
        input_path,
        [
            _row(
                "budget",
                label="review",
                reason="model_accept_clamped_low_confidence:0.51",
                risk_tags=[],
                text="candidate text for later local judge",
                local_judge_action="skipped_by_budget",
            )
        ],
    )

    result = build_review_worklist(input_path=input_path, output_path=output_path, include_text=True)

    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported["recommended_action"] == "defer_local_judge_until_budget_available"
    assert result.summary["recommended_action_counts"] == {"defer_local_judge_until_budget_available": 1}


def test_review_worklist_cli_writes_outputs(tmp_path) -> None:
    input_path = tmp_path / "labeled.jsonl"
    output_path = tmp_path / "worklist.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_jsonl(
        input_path,
        [_row("low", label="review", reason="low_confidence", risk_tags=[])],
    )

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary",
            str(summary_path),
            "--max-rows",
            "1",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["max_rows"] == 1
    assert summary["review_candidate_count"] == 1


def _row(
    candidate_id: str,
    *,
    label: str,
    reason: str,
    risk_tags: list[str],
    text: str | None = None,
    char_count: int = 100,
    local_judge_action: str | None = None,
) -> dict:
    row = {
        "schema_version": "prefix-semantic-candidate-v1",
        "candidate_id": candidate_id,
        "source_path": "source.py",
        "symbol": "SYSTEM_MESSAGE",
        "parent_hash": "parent",
        "text_hash": f"text-{candidate_id}",
        "semantic_hint": "team_policy",
        "confidence": 0.7,
        "char_count": char_count,
        "line_count": 1,
        "risk_tags": risk_tags,
        "label": label,
        "label_source": "conservative_rule_judge_v1",
        "label_reason": reason,
    }
    if text is not None:
        row["text"] = text
    if local_judge_action is not None:
        row["local_judge_action"] = local_judge_action
    return row


def _write_jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
