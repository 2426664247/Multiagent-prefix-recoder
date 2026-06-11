from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .candidate_label_eval import ALLOWED_LABELS


@dataclass(frozen=True)
class LocalJudgeGoldsetValidationResult:
    input_path: str
    summary_path: str | None
    summary: dict[str, Any]


def validate_local_judge_goldset(
    *,
    input_path: str | Path,
    summary_path: str | Path | None = None,
    min_samples: int = 20,
    min_label_count: int = 1,
) -> LocalJudgeGoldsetValidationResult:
    source_path = Path(input_path)
    rows = tuple(_load_jsonl(source_path))
    items = tuple(_validate_row(row, index=index) for index, row in enumerate(rows))
    summary = _summary(
        input_path=str(source_path),
        items=items,
        min_samples=max(0, min_samples),
        min_label_count=max(0, min_label_count),
    )
    summary_target = _write_json(summary_path, summary)
    return LocalJudgeGoldsetValidationResult(
        input_path=str(source_path),
        summary_path=str(summary_target) if summary_target else None,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a local gold-label JSONL before running local_judge_quality_eval."
    )
    parser.add_argument("--input", required=True, help="Local gold JSONL with text and expected_label/gold_label.")
    parser.add_argument("--summary", help="Optional prompt-safe validation summary JSON.")
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-label-count", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_local_judge_goldset(
        input_path=args.input,
        summary_path=args.summary,
        min_samples=args.min_samples,
        min_label_count=args.min_label_count,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0 if result.summary.get("ready") is True else 1


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if isinstance(value, dict):
                yield value


def _validate_row(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    text = row.get("text")
    expected = str(row.get("expected_label") or row.get("gold_label") or "").strip().lower()
    row_errors: list[str] = []
    if expected not in ALLOWED_LABELS:
        row_errors.append("invalid_expected_label")
    if not isinstance(text, str) or not text.strip():
        row_errors.append("missing_text")
    risk_tags = row.get("risk_tags") if isinstance(row.get("risk_tags"), list) else []
    return {
        "row_index": index,
        "example_id": _safe_id(row.get("id") or row.get("candidate_id") or f"row-{index}"),
        "expected_label": expected if expected in ALLOWED_LABELS else None,
        "text_hash": _text_hash(text if isinstance(text, str) else ""),
        "has_text": isinstance(text, str) and bool(text.strip()),
        "char_count": len(text) if isinstance(text, str) else 0,
        "line_count": text.count("\n") + 1 if isinstance(text, str) and text else 0,
        "semantic_hint": row.get("semantic_hint"),
        "risk_tag_count": len(risk_tags),
        "errors": tuple(row_errors),
        "valid": not row_errors,
    }


def _summary(
    *,
    input_path: str,
    items: Sequence[Mapping[str, Any]],
    min_samples: int,
    min_label_count: int,
) -> dict[str, Any]:
    valid = tuple(item for item in items if item.get("valid") is True)
    invalid = tuple(item for item in items if item.get("valid") is not True)
    expected_counts = Counter(str(item.get("expected_label")) for item in valid if item.get("expected_label"))
    error_counts: Counter[str] = Counter()
    for item in invalid:
        error_counts.update(str(error) for error in item.get("errors") or ())
    labels_with_min = sum(1 for count in expected_counts.values() if count >= min_label_count)
    ready = bool(
        len(valid) >= min_samples
        and not invalid
        and labels_with_min >= len(ALLOWED_LABELS)
    )
    return {
        "schema_version": "prefix-local-judge-goldset-validation-summary-v1",
        "prompt_safe_summary": True,
        "input_path": input_path,
        "min_samples": min_samples,
        "min_label_count": min_label_count,
        "row_count": len(items),
        "valid_row_count": len(valid),
        "invalid_row_count": len(invalid),
        "missing_text_count": error_counts.get("missing_text", 0),
        "invalid_expected_label_count": error_counts.get("invalid_expected_label", 0),
        "expected_label_counts": dict(sorted(expected_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "labels_with_min_count": labels_with_min,
        "label_options": sorted(ALLOWED_LABELS),
        "gold_text_written": False,
        "real_provider_metrics_available": False,
        "ready": ready,
        "ready_for_local_judge_quality_eval": ready,
        "recommendation": _recommendation(
            ready=ready,
            valid_count=len(valid),
            min_samples=min_samples,
            invalid_count=len(invalid),
            labels_with_min_count=labels_with_min,
        ),
        "limits": (
            "This validates local gold JSONL structure only. It does not call a local judge, "
            "provider API, or prove semantic quality, cache, latency, cost, or task success."
        ),
    }


def _recommendation(
    *,
    ready: bool,
    valid_count: int,
    min_samples: int,
    invalid_count: int,
    labels_with_min_count: int,
) -> str:
    if ready:
        return "run_local_judge_quality_eval"
    if invalid_count > 0:
        return "fix_missing_text_or_expected_labels"
    if valid_count < min_samples:
        return "add_more_gold_labeled_rows"
    if labels_with_min_count < len(ALLOWED_LABELS):
        return "add_label_balance_before_quality_eval"
    return "inspect_goldset_validation_summary"


def _text_hash(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_id(value: Any) -> str:
    text = str(value).strip()
    return text[:160] or "gold-row"


def _write_json(path: str | Path | None, value: Mapping[str, Any]) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return target


if __name__ == "__main__":
    raise SystemExit(main())
