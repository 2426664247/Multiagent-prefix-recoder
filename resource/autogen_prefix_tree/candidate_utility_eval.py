from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CandidateUtilityEvalResult:
    input_path: str
    summary_path: str | None
    summary: dict[str, Any]


def evaluate_candidate_utility(
    *,
    input_path: str | Path,
    summary_path: str | Path | None = None,
    include_review: bool = False,
) -> CandidateUtilityEvalResult:
    source_path = Path(input_path)
    rows = tuple(_load_jsonl(source_path))
    eligible_rows = tuple(_eligible_rows(rows, include_review=include_review))
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible_rows:
        groups[str(row.get("text_hash") or row.get("candidate_id") or "")].append(row)
    repeated_groups = {key: value for key, value in groups.items() if key and len(value) >= 2}
    summary = _summarize(rows, eligible_rows, repeated_groups, input_path=str(source_path), include_review=include_review)
    summary_target = _write_json(summary_path, summary)
    return CandidateUtilityEvalResult(
        input_path=str(source_path),
        summary_path=str(summary_target) if summary_target else None,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate prompt-safe character utility from labeled semantic candidates."
    )
    parser.add_argument("--input", required=True, help="Input labeled semantic candidate JSONL.")
    parser.add_argument("--summary", help="Optional utility summary JSON output.")
    parser.add_argument("--include-review", action="store_true", help="Include review labels as potential candidates.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_candidate_utility(
        input_path=args.input,
        summary_path=args.summary,
        include_review=args.include_review,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if isinstance(value, dict):
                yield value


def _eligible_rows(rows: Sequence[Mapping[str, Any]], *, include_review: bool) -> Iterable[Mapping[str, Any]]:
    allowed = {"accept", "accepted"}
    if include_review:
        allowed.add("review")
    for row in rows:
        if str(row.get("label") or "").lower() in allowed:
            yield row


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    eligible_rows: Sequence[Mapping[str, Any]],
    repeated_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    input_path: str,
    include_review: bool,
) -> dict[str, Any]:
    label_counts = Counter(str(row.get("label") or "missing") for row in rows)
    eligible_hint_counts = Counter(str(row.get("semantic_hint") or "unknown") for row in eligible_rows)
    eligible_parent_hashes = {str(row.get("parent_hash") or "") for row in eligible_rows if row.get("parent_hash")}
    eligible_sources = {str(row.get("source_path") or "") for row in eligible_rows if row.get("source_path")}
    eligible_chars = sum(_int(row.get("char_count")) for row in eligible_rows)
    prompt_safe_review_count = sum(1 for row in eligible_rows if str(row.get("label") or "").lower() == "review")
    repeated_hint_counts: Counter[str] = Counter()
    repeated_parent_hashes: set[str] = set()
    repeated_sources: set[str] = set()
    total_repeated_candidate_chars = 0
    total_estimated_reusable_chars = 0
    top_groups: list[dict[str, Any]] = []
    for text_hash, group in repeated_groups.items():
        first = group[0]
        char_count = _int(first.get("char_count"))
        count = len(group)
        semantic_hint = str(first.get("semantic_hint") or "unknown")
        repeated_hint_counts[semantic_hint] += count
        repeated_parent_hashes.update(str(row.get("parent_hash") or "") for row in group if row.get("parent_hash"))
        repeated_sources.update(str(row.get("source_path") or "") for row in group if row.get("source_path"))
        total_repeated_candidate_chars += char_count * count
        total_estimated_reusable_chars += char_count * (count - 1)
        top_groups.append(
            {
                "text_hash": text_hash,
                "semantic_hint": semantic_hint,
                "count": count,
                "char_count": char_count,
                "estimated_reusable_chars": char_count * (count - 1),
                "labels": sorted({str(row.get("label") or "missing") for row in group}),
                "risk_tags": sorted({str(tag) for row in group for tag in row.get("risk_tags") or ()}),
            }
        )
    top_groups.sort(key=lambda row: (-int(row["estimated_reusable_chars"]), str(row["text_hash"])))
    return {
        "schema_version": "prefix-semantic-candidate-utility-summary-v1",
        "input_path": input_path,
        "candidate_count": len(rows),
        "include_review": include_review,
        "promotion_policy": _promotion_policy(include_review=include_review),
        "label_counts": dict(sorted(label_counts.items())),
        "eligible_candidate_count": len(eligible_rows),
        "eligible_parent_block_count": len(eligible_parent_hashes),
        "eligible_source_file_count": len(eligible_sources),
        "eligible_candidate_chars": eligible_chars,
        "eligible_review_candidate_count": prompt_safe_review_count,
        "eligible_semantic_hint_counts": dict(sorted(eligible_hint_counts.items())),
        "repeated_candidate_group_count": len(repeated_groups),
        "repeated_candidate_occurrence_count": sum(len(group) for group in repeated_groups.values()),
        "repeated_semantic_hint_counts": dict(sorted(repeated_hint_counts.items())),
        "repeated_parent_block_count": len(repeated_parent_hashes),
        "repeated_source_file_count": len(repeated_sources),
        "total_repeated_candidate_chars": total_repeated_candidate_chars,
        "total_estimated_reusable_chars": total_estimated_reusable_chars,
        "top_repeated_candidate_groups": top_groups[:20],
    }


def _promotion_policy(*, include_review: bool) -> dict[str, Any]:
    if include_review:
        return {
            "schema_version": "prefix-candidate-promotion-policy-v1",
            "automatic_promotion_allowed_labels": ["accept"],
            "review_candidate_policy": "upper_bound_only_require_local_judge_or_manual_review",
            "review_candidates_used_for_upper_bound_only": True,
            "automatic_promotion_includes_review": False,
        }
    return {
        "schema_version": "prefix-candidate-promotion-policy-v1",
        "automatic_promotion_allowed_labels": ["accept"],
        "review_candidate_policy": "excluded_from_automatic_promotion",
        "review_candidates_used_for_upper_bound_only": False,
        "automatic_promotion_includes_review": False,
    }


def _write_json(path: str | Path | None, value: Mapping[str, Any]) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
