from __future__ import annotations

import json

from autogen_prefix_tree.ir import stable_hash
from autogen_prefix_tree.local_judge_goldset_builder import build_local_judge_goldset_template, main


def test_goldset_template_is_prompt_safe_by_default(tmp_path) -> None:
    input_path = tmp_path / "semantic_candidates_labeled.jsonl"
    output_path = tmp_path / "goldset_template.jsonl"
    summary_path = tmp_path / "goldset_summary.json"
    _write_jsonl(
        input_path,
        [
            _row("accept-1", label="accept", text="Stable verification prompt text."),
            _row("review-1", label="review", text="Agent identity prompt text.", risk_tags=["agent_identity_boundary"]),
        ],
    )

    result = build_local_judge_goldset_template(
        input_paths=[input_path],
        output_path=output_path,
        summary_path=summary_path,
        max_rows=2,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    serialized_rows = json.dumps(rows, ensure_ascii=False)
    serialized_summary = json.dumps(result.summary, ensure_ascii=False)
    assert [row["expected_label"] for row in rows] == [None, None]
    assert all(row["label_options"] == ["accept", "reject", "review"] for row in rows)
    assert all("text" not in row for row in rows)
    assert "Stable verification prompt text" not in serialized_rows
    assert "Agent identity prompt text" not in serialized_rows
    assert "Stable verification prompt text" not in serialized_summary
    assert result.summary["prompt_safe_summary"] is True
    assert result.summary["gold_text_written"] is False
    assert result.summary["ready_for_local_judge_quality_eval"] is False
    assert result.summary["manual_labeling_required"] is True
    assert result.summary["expected_label_pending_count"] == 2
    assert result.summary["source_label_counts"] == {"accept": 1, "review": 1}
    assert result.summary["risk_tag_counts"] == {"agent_identity_boundary": 1}
    assert json.loads(summary_path.read_text(encoding="utf-8"))["schema_version"] == (
        "prefix-local-judge-goldset-template-summary-v1"
    )


def test_goldset_template_can_explicitly_include_text_for_annotation(tmp_path) -> None:
    input_path = tmp_path / "review_worklist.jsonl"
    output_path = tmp_path / "goldset_with_text.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                **_row("review-1", label="review", text="Candidate text for human label."),
                "schema_version": "prefix-review-worklist-item-v1",
                "review_priority": "high",
            },
            {
                **_row("missing-text", label="review", text=None),
                "schema_version": "prefix-review-worklist-item-v1",
            },
        ],
    )

    result = build_local_judge_goldset_template(
        input_paths=[input_path],
        output_path=output_path,
        include_text=True,
        require_text=True,
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["candidate_id"] == "review-1"
    assert row["text"] == "Candidate text for human label."
    assert row["expected_label"] is None
    assert row["source_schema_version"] == "prefix-review-worklist-item-v1"
    assert row["review_priority"] == "high"
    assert result.summary["include_text"] is True
    assert result.summary["gold_text_written"] is True
    assert result.summary["selected_missing_text_count"] == 0
    assert result.summary["recommendation"] == "fill_expected_label_then_run_local_judge_quality_eval"


def test_goldset_template_recovers_candidate_text_from_source_prompts(tmp_path) -> None:
    candidate_path = tmp_path / "semantic_candidates_labeled.jsonl"
    source_prompts_path = tmp_path / "source_prompts.jsonl"
    output_path = tmp_path / "goldset_with_recovered_text.local.jsonl"
    summary_path = tmp_path / "goldset_summary.json"
    prompt = "Verify evidence carefully before final answer.\n\nNo prompt text should appear in summaries."
    segment = "Verify evidence carefully before final answer."
    parent_hash = stable_hash({"text": prompt})
    text_hash = stable_hash({"semantic_candidate": segment})
    _write_jsonl(source_prompts_path, [_source_prompt_row(prompt)])
    _write_jsonl(
        candidate_path,
        [
            {
                **_row("candidate-1", label="review", text=None),
                "parent_hash": parent_hash,
                "text_hash": text_hash,
            }
        ],
    )

    result = build_local_judge_goldset_template(
        input_paths=[candidate_path],
        source_prompt_paths=[source_prompts_path],
        output_path=output_path,
        summary_path=summary_path,
        include_text=True,
        require_text=True,
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    serialized_summary = json.dumps(result.summary, ensure_ascii=False)
    assert row["text"] == segment
    assert row["text_source"] == "source_prompts"
    assert row["text_recovered"] is True
    assert row["text_recovery_source_prompt_path"] == str(source_prompts_path)
    assert result.summary["selected_count"] == 1
    assert result.summary["selected_with_text_count"] == 1
    assert result.summary["selected_recovered_text_count"] == 1
    assert result.summary["selected_missing_text_count"] == 0
    assert result.summary["text_recovery"]["enabled"] is True
    assert result.summary["text_recovery"]["indexed_prompt_count"] == 1
    assert result.summary["text_recovery"]["indexed_segment_count"] == 2
    assert result.summary["text_recovery"]["selected_recovered_text_count"] == 1
    assert segment not in serialized_summary
    assert "No prompt text should appear in summaries." not in serialized_summary
    assert json.loads(summary_path.read_text(encoding="utf-8"))["gold_text_written"] is True


def test_goldset_template_does_not_recover_text_without_source_prompts(tmp_path) -> None:
    candidate_path = tmp_path / "semantic_candidates_labeled.jsonl"
    output_path = tmp_path / "goldset_empty.jsonl"
    _write_jsonl(candidate_path, [_row("candidate-1", label="review", text=None)])

    result = build_local_judge_goldset_template(
        input_paths=[candidate_path],
        output_path=output_path,
        include_text=True,
        require_text=True,
    )

    assert output_path.read_text(encoding="utf-8") == ""
    assert result.summary["selected_count"] == 0
    assert result.summary["selected_missing_text_count"] == 0
    assert result.summary["text_recovery"]["enabled"] is False
    assert result.summary["recommendation"] == "add_real_semantic_candidate_inputs_before_gold_labeling"


def test_goldset_template_stratified_sampling_is_deterministic(tmp_path) -> None:
    input_path = tmp_path / "semantic_candidates_labeled.jsonl"
    rows = []
    for index in range(10):
        label = "accept" if index % 2 == 0 else "review"
        hint = "verification_policy" if index < 5 else "team_policy"
        rows.append(_row(f"candidate-{index}", label=label, semantic_hint=hint, text=f"text {index}"))
    _write_jsonl(input_path, rows)

    first = build_local_judge_goldset_template(
        input_paths=[input_path],
        output_path=tmp_path / "first.jsonl",
        max_rows=5,
        seed=7,
    )
    second = build_local_judge_goldset_template(
        input_paths=[input_path],
        output_path=tmp_path / "second.jsonl",
        max_rows=5,
        seed=7,
    )

    first_rows = [json.loads(line) for line in (tmp_path / "first.jsonl").read_text(encoding="utf-8").splitlines()]
    second_rows = [json.loads(line) for line in (tmp_path / "second.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["candidate_id"] for row in first_rows] == [row["candidate_id"] for row in second_rows]
    assert first.summary["selected_count"] == 5
    assert set(first.summary["source_label_counts"]) == {"accept", "review"}
    assert set(first.summary["semantic_hint_counts"]) == {"team_policy", "verification_policy"}


def test_goldset_template_cli_writes_outputs(tmp_path) -> None:
    input_path = tmp_path / "semantic_candidates_labeled.jsonl"
    output_path = tmp_path / "goldset.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_jsonl(input_path, [_row("candidate-1", label="review", text="candidate text")])

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary",
            str(summary_path),
            "--label",
            "review",
            "--max-rows",
            "1",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["selected_count"] == 1
    assert summary["filters"]["labels"] == ["review"]


def test_goldset_template_cli_accepts_source_prompts_for_local_text_recovery(tmp_path) -> None:
    candidate_path = tmp_path / "semantic_candidates_labeled.jsonl"
    source_prompts_path = tmp_path / "source_prompts.jsonl"
    output_path = tmp_path / "goldset.jsonl"
    summary_path = tmp_path / "summary.json"
    prompt = "Check the evidence before finalizing the answer."
    parent_hash = stable_hash({"text": prompt})
    text_hash = stable_hash({"semantic_candidate": prompt})
    _write_jsonl(source_prompts_path, [_source_prompt_row(prompt)])
    _write_jsonl(
        candidate_path,
        [
            {
                **_row("candidate-1", label="review", text=None),
                "parent_hash": parent_hash,
                "text_hash": text_hash,
            }
        ],
    )

    exit_code = main(
        [
            "--input",
            str(candidate_path),
            "--source-prompts",
            str(source_prompts_path),
            "--output",
            str(output_path),
            "--summary",
            str(summary_path),
            "--include-text",
            "--require-text",
        ]
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert row["text"] == prompt
    assert row["text_recovered"] is True
    assert summary["text_recovery"]["selected_recovered_text_count"] == 1
    assert prompt not in json.dumps(summary, ensure_ascii=False)


def _row(
    candidate_id: str,
    *,
    label: str,
    text: str | None,
    semantic_hint: str = "verification_policy",
    risk_tags: list[str] | None = None,
) -> dict:
    row = {
        "schema_version": "prefix-semantic-candidate-v1",
        "candidate_id": candidate_id,
        "source_path": "source.py",
        "symbol": "SYSTEM_MESSAGE",
        "extraction": "prompt_named_constant",
        "parent_hash": f"parent-{candidate_id}",
        "text_hash": f"text-{candidate_id}",
        "semantic_hint": semantic_hint,
        "confidence": 0.74,
        "char_count": 120,
        "line_count": 2,
        "risk_tags": risk_tags or [],
        "label": label,
        "label_source": "conservative_rule_judge_v1",
        "label_reason": "accepted_by_conservative_rule_judge" if label == "accept" else "low_confidence",
        "rule_label": label,
        "rule_label_reason": "rule reason",
    }
    if text is not None:
        row["text"] = text
    return row


def _source_prompt_row(prompt: str) -> dict:
    return {
        "body": {
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Evaluate this prompt structure for offline prefix coverage."},
            ],
            "source_prompt_metadata": {"content_hash": stable_hash({"text": prompt})},
        },
        "id": "source-prompt:000001",
        "session_id": "source-prompt",
    }


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
