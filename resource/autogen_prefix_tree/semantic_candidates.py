from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .ir import stable_hash


@dataclass(frozen=True)
class SemanticCandidate:
    candidate_id: str
    parent_hash: str
    semantic_hint: str
    confidence: float
    char_count: int
    line_count: int
    text_hash: str
    risk_tags: tuple[str, ...]


@dataclass(frozen=True)
class SemanticCandidateSummary:
    candidate_count: int
    parent_block_count: int
    semantic_hint_counts: dict[str, int]
    risk_tag_counts: dict[str, int]
    top_candidates: tuple[dict[str, object], ...]


def extract_semantic_candidates(text: str, *, parent_hash: str) -> tuple[SemanticCandidate, ...]:
    candidates: list[SemanticCandidate] = []
    for index, segment in enumerate(_segments(text), start=1):
        semantic_hint, confidence, risk_tags = _classify_segment(segment)
        if semantic_hint is None:
            continue
        text_hash = stable_hash({"semantic_candidate": segment})
        candidates.append(
            SemanticCandidate(
                candidate_id=f"{parent_hash[:12]}:{index:03d}:{text_hash[:8]}",
                parent_hash=parent_hash,
                semantic_hint=semantic_hint,
                confidence=confidence,
                char_count=len(segment),
                line_count=max(1, segment.count("\n") + 1),
                text_hash=text_hash,
                risk_tags=risk_tags,
            )
        )
    return tuple(candidates)


def summarize_semantic_candidates(candidates: Sequence[SemanticCandidate]) -> SemanticCandidateSummary:
    semantic_counts = Counter(candidate.semantic_hint for candidate in candidates)
    risk_counts: Counter[str] = Counter()
    for candidate in candidates:
        risk_counts.update(candidate.risk_tags)
    top = sorted(
        candidates,
        key=lambda candidate: (-candidate.confidence, -candidate.char_count, candidate.text_hash),
    )[:20]
    return SemanticCandidateSummary(
        candidate_count=len(candidates),
        parent_block_count=len({candidate.parent_hash for candidate in candidates}),
        semantic_hint_counts=dict(sorted(semantic_counts.items())),
        risk_tag_counts=dict(sorted(risk_counts.items())),
        top_candidates=tuple(_candidate_metadata(candidate) for candidate in top),
    )


def semantic_candidate_summary_to_dict(summary: SemanticCandidateSummary) -> dict[str, object]:
    return {
        "candidate_count": summary.candidate_count,
        "parent_block_count": summary.parent_block_count,
        "semantic_hint_counts": summary.semantic_hint_counts,
        "risk_tag_counts": summary.risk_tag_counts,
        "top_candidates": summary.top_candidates,
    }


def _candidate_metadata(candidate: SemanticCandidate) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "parent_hash": candidate.parent_hash,
        "semantic_hint": candidate.semantic_hint,
        "confidence": candidate.confidence,
        "char_count": candidate.char_count,
        "line_count": candidate.line_count,
        "text_hash": candidate.text_hash,
        "risk_tags": candidate.risk_tags,
    }


def _segments(text: str) -> Iterable[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(paragraph) <= 220 or len(lines) <= 1:
            yield paragraph
            continue
        for line in lines:
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if cleaned:
                yield cleaned


def _classify_segment(segment: str) -> tuple[str | None, float, tuple[str, ...]]:
    lowered = segment.lower()
    risk_tags: set[str] = set()
    if re.search(r"\byou are\b", lowered):
        risk_tags.add("agent_identity_boundary")
    if "user" in lowered and ("cannot" in lowered or "can't" in lowered or "must" in lowered):
        risk_tags.add("user_interaction_constraint")
    if "if " in lowered or "when " in lowered:
        risk_tags.add("conditional_instruction")
    if "current" in lowered or "latest" in lowered:
        risk_tags.add("possible_turn_specific_reference")

    if any(marker in lowered for marker in ("verify", "evidence", "check the", "check if", "confirmed")):
        return "verification_policy", 0.74, tuple(sorted(risk_tags))
    if any(marker in lowered for marker in ("tool", "function", "code block", "script", "execute", "browser")):
        return "tool_or_code_policy", 0.72, tuple(sorted(risk_tags))
    if any(marker in lowered for marker in ("do not", "don't", "must", "cannot", "can't", "only return", "reply with")):
        return "team_policy", 0.66, tuple(sorted(risk_tags))
    if any(marker in lowered for marker in ("step by step", "next step", "progress", "plan", "select the next")):
        return "procedure_policy", 0.6, tuple(sorted(risk_tags))
    if len(segment) >= 300:
        return "long_stable_instruction", 0.52, tuple(sorted(risk_tags))
    return None, 0.0, tuple(sorted(risk_tags))
