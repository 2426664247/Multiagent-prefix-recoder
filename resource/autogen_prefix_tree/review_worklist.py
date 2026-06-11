from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HIGH_PRIORITY_REASONS = (
    "high_risk_boundary",
    "local_judge_error",
    "missing_candidate_text",
    "model_accept_clamped",
)


@dataclass(frozen=True)
class ReviewWorklistResult:
    input_path: str
    output_path: str
    summary_path: str | None
    summary: dict[str, Any]


def build_review_worklist(
    *,
    input_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    include_text: bool = False,
    max_rows: int | None = None,
) -> ReviewWorklistResult:
    source_path = Path(input_path)
    rows = tuple(_review_rows(_load_jsonl(source_path), include_text=include_text))
    sorted_rows = tuple(sorted(rows, key=_sort_key))
    if max_rows is not None:
        sorted_rows = sorted_rows[: max(0, max_rows)]
    output_target = Path(output_path)
    _write_jsonl(output_target, sorted_rows)
    summary = _summary(
        sorted_rows,
        input_path=str(source_path),
        output_path=str(output_target),
        include_text=include_text,
        max_rows=max_rows,
    )
    summary_target = _write_json(Path(summary_path), summary) if summary_path is not None else None
    return ReviewWorklistResult(
        input_path=str(source_path),
        output_path=str(output_target),
        summary_path=str(summary_target) if summary_target else None,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a prioritized review worklist from labeled semantic candidates."
    )
    parser.add_argument("--input", required=True, help="Input semantic_candidates_labeled.jsonl.")
    parser.add_argument("--output", required=True, help="Output review worklist JSONL.")
    parser.add_argument("--summary", help="Optional prompt-safe worklist summary JSON.")
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include raw candidate text in the worklist. Default is prompt-safe metadata only.",
    )
    parser.add_argument("--max-rows", type=int, help="Optional maximum number of review rows to export.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_review_worklist(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary,
        include_text=args.include_text,
        max_rows=args.max_rows,
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


def _review_rows(rows: Iterable[Mapping[str, Any]], *, include_text: bool) -> Iterable[dict[str, Any]]:
    for row in rows:
        if str(row.get("label") or "").lower() != "review":
            continue
        item = {
            "schema_version": "prefix-review-worklist-item-v1",
            "candidate_id": row.get("candidate_id"),
            "source_path": row.get("source_path"),
            "symbol": row.get("symbol"),
            "parent_hash": row.get("parent_hash"),
            "text_hash": row.get("text_hash"),
            "semantic_hint": row.get("semantic_hint"),
            "classifier_confidence": row.get("confidence"),
            "char_count": row.get("char_count"),
            "line_count": row.get("line_count"),
            "risk_tags": row.get("risk_tags") or [],
            "label_reason": row.get("label_reason"),
            "label_source": row.get("label_source"),
            "local_judge_action": row.get("local_judge_action"),
            "rule_label": row.get("rule_label"),
            "rule_label_reason": row.get("rule_label_reason"),
            "model_label": row.get("model_label"),
            "model_confidence": row.get("model_confidence"),
            "review_priority": _priority(row),
            "recommended_action": _recommended_action(row, include_text=include_text),
        }
        if include_text and isinstance(row.get("text"), str):
            item["text"] = row.get("text")
        yield item


def _priority(row: Mapping[str, Any]) -> str:
    reason = str(row.get("label_reason") or "")
    risk_tags = {str(tag) for tag in row.get("risk_tags") or ()}
    if any(reason.startswith(marker) for marker in HIGH_PRIORITY_REASONS):
        return "high"
    if risk_tags:
        return "high"
    if reason == "low_confidence":
        return "medium"
    return "low"


def _recommended_action(row: Mapping[str, Any], *, include_text: bool) -> str:
    reason = str(row.get("label_reason") or "")
    local_judge_action = str(row.get("local_judge_action") or "")
    if reason == "missing_candidate_text" or local_judge_action == "missing_candidate_text":
        return "rerun_worklist_with_include_text_for_local_judge"
    if local_judge_action == "local_judge_error":
        return "retry_local_judge_or_human_review"
    if local_judge_action == "skipped_by_budget":
        return "defer_local_judge_until_budget_available"
    if local_judge_action == "model_called":
        if reason.startswith("model_accept_clamped"):
            return "review_model_accept_static_clamp"
        return "human_review_local_judge_result"
    if not include_text:
        return "rerun_worklist_with_include_text_for_local_judge"
    if _priority(row) == "high":
        return "local_judge_with_static_safety_clamps"
    return "batch_local_judge_or_human_spot_check"


def _sort_key(row: Mapping[str, Any]) -> tuple[int, int, str, str]:
    priority_order = {"high": 0, "medium": 1, "low": 2}
    return (
        priority_order.get(str(row.get("review_priority")), 9),
        -_int(row.get("char_count")),
        str(row.get("source_path") or ""),
        str(row.get("candidate_id") or ""),
    )


def _summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    input_path: str,
    output_path: str,
    include_text: bool,
    max_rows: int | None,
) -> dict[str, Any]:
    priority_counts = Counter(str(row.get("review_priority") or "unknown") for row in rows)
    reason_counts = Counter(str(row.get("label_reason") or "missing") for row in rows)
    hint_counts = Counter(str(row.get("semantic_hint") or "unknown") for row in rows)
    action_counts = Counter(str(row.get("recommended_action") or "unknown") for row in rows)
    local_judge_action_counts = Counter(str(row.get("local_judge_action") or "missing") for row in rows)
    risk_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for row in rows:
        risk_counts.update(str(tag) for tag in row.get("risk_tags") or ())
        if row.get("source_path"):
            source_counts[str(row.get("source_path"))] += 1
    return {
        "schema_version": "prefix-review-worklist-summary-v1",
        "input_path": input_path,
        "output_path": output_path,
        "include_text": include_text,
        "prompt_safe_summary": True,
        "max_rows": max_rows,
        "review_candidate_count": len(rows),
        "priority_counts": dict(sorted(priority_counts.items())),
        "recommended_action_counts": dict(sorted(action_counts.items())),
        "local_judge_action_counts": dict(sorted(local_judge_action_counts.items())),
        "semantic_hint_counts": dict(sorted(hint_counts.items())),
        "risk_tag_counts": dict(sorted(risk_counts.items())),
        "label_reason_counts": dict(sorted(reason_counts.items())),
        "top_source_files": [
            {"source_path": source, "review_candidate_count": count}
            for source, count in source_counts.most_common(10)
        ],
        "text_artifact_note": (
            "output JSONL contains raw candidate text; keep it local and do not commit"
            if include_text
            else "output JSONL is prompt-safe metadata only"
        ),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
