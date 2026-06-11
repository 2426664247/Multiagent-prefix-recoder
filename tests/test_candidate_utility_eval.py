from __future__ import annotations

import json

from autogen_prefix_tree.candidate_utility_eval import evaluate_candidate_utility


def test_candidate_utility_eval_counts_only_repeated_accepted_candidates_by_default(tmp_path) -> None:
    input_path = tmp_path / "labeled.jsonl"
    summary_path = tmp_path / "summary.json"
    rows = [
        _row("a1", text_hash="same", label="accept", semantic_hint="team_policy", char_count=100),
        _row("a2", text_hash="same", label="accept", semantic_hint="team_policy", char_count=100),
        _row("b1", text_hash="single", label="accept", semantic_hint="tool_or_code_policy", char_count=200),
        _row("c1", text_hash="reviewed", label="review", semantic_hint="verification_policy", char_count=300),
        _row("c2", text_hash="reviewed", label="review", semantic_hint="verification_policy", char_count=300),
    ]
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = evaluate_candidate_utility(input_path=input_path, summary_path=summary_path)

    assert result.summary["eligible_candidate_count"] == 3
    assert result.summary["eligible_parent_block_count"] == 3
    assert result.summary["eligible_source_file_count"] == 1
    assert result.summary["eligible_candidate_chars"] == 400
    assert result.summary["eligible_review_candidate_count"] == 0
    assert result.summary["promotion_policy"] == {
        "schema_version": "prefix-candidate-promotion-policy-v1",
        "automatic_promotion_allowed_labels": ["accept"],
        "review_candidate_policy": "excluded_from_automatic_promotion",
        "review_candidates_used_for_upper_bound_only": False,
        "automatic_promotion_includes_review": False,
    }
    assert result.summary["repeated_candidate_group_count"] == 1
    assert result.summary["total_estimated_reusable_chars"] == 100
    assert result.summary["repeated_semantic_hint_counts"] == {"team_policy": 2}
    assert result.summary["top_repeated_candidate_groups"][0]["text_hash"] == "same"
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["total_repeated_candidate_chars"] == 200


def test_candidate_utility_eval_can_include_review_candidates(tmp_path) -> None:
    input_path = tmp_path / "labeled.jsonl"
    rows = [
        _row("a1", text_hash="reviewed", label="review", semantic_hint="verification_policy", char_count=300),
        _row("a2", text_hash="reviewed", label="review", semantic_hint="verification_policy", char_count=300),
    ]
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = evaluate_candidate_utility(input_path=input_path, include_review=True)

    assert result.summary["eligible_candidate_count"] == 2
    assert result.summary["eligible_parent_block_count"] == 2
    assert result.summary["eligible_review_candidate_count"] == 2
    assert result.summary["promotion_policy"] == {
        "schema_version": "prefix-candidate-promotion-policy-v1",
        "automatic_promotion_allowed_labels": ["accept"],
        "review_candidate_policy": "upper_bound_only_require_local_judge_or_manual_review",
        "review_candidates_used_for_upper_bound_only": True,
        "automatic_promotion_includes_review": False,
    }
    assert result.summary["repeated_candidate_group_count"] == 1
    assert result.summary["total_estimated_reusable_chars"] == 300


def _row(candidate_id: str, *, text_hash: str, label: str, semantic_hint: str, char_count: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "parent_hash": f"parent-{candidate_id}",
        "source_path": "source.py",
        "text_hash": text_hash,
        "label": label,
        "semantic_hint": semantic_hint,
        "char_count": char_count,
        "risk_tags": [],
    }
