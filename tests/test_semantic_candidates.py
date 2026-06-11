from __future__ import annotations

from autogen_prefix_tree.ir import stable_hash
from autogen_prefix_tree.semantic_candidates import extract_semantic_candidates, summarize_semantic_candidates


def test_extract_semantic_candidates_classifies_policy_like_segments_without_text() -> None:
    text = """You are a helpful AI assistant.
When using code, you must indicate the script type in the code block.
Do not ask users to copy and paste the result.
When you find an answer, verify the answer carefully and include evidence."""
    parent_hash = stable_hash({"text": text})

    candidates = extract_semantic_candidates(text, parent_hash=parent_hash)
    summary = summarize_semantic_candidates(candidates)

    assert summary.candidate_count >= 3
    assert summary.semantic_hint_counts["tool_or_code_policy"] >= 1
    assert summary.semantic_hint_counts["verification_policy"] >= 1
    assert summary.semantic_hint_counts["team_policy"] >= 1
    top_as_text = repr(summary.top_candidates)
    assert "copy and paste" not in top_as_text
    assert "verify the answer" not in top_as_text
