from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .candidate_label_eval import (
    ALLOWED_LABELS,
    HIGH_RISK_TAGS,
    LocalModelJudgeConfig,
    _label_row_by_local_model,
    _label_row_by_rules,
)


CSV_COLUMNS = (
    "id",
    "expected_label",
    "annotation_notes",
    "label_options",
    "semantic_hint",
    "risk_tags",
    "source_label",
    "rule_label",
    "model_label",
    "review_priority",
    "recommended_action",
    "source_path",
    "symbol",
    "extraction",
    "parent_hash",
    "text_hash",
    "text",
)

SUGGESTION_REVIEW_COLUMNS = (
    "id",
    "expected_label",
    "suggested_label",
    "annotation_notes",
    "suggestion_notes",
    "source_label",
    "rule_label",
    "model_label",
    "model_confidence",
    "confidence_meets_min",
    "static_safety_clamped",
    "local_judge_action",
    "label_reason_code",
    "semantic_hint",
    "risk_tags",
    "review_priority",
    "recommended_action",
    "source_path",
    "symbol",
    "extraction",
    "parent_hash",
    "text_hash",
    "text",
)


@dataclass(frozen=True)
class GoldsetCsvResult:
    input_path: str
    output_path: str
    summary_path: str | None
    summary: dict[str, Any]


@dataclass(frozen=True)
class GoldsetCsvProgressResult:
    input_path: str
    summary_path: str | None
    guide_path: str | None
    worklist_path: str | None
    summary: dict[str, Any]


@dataclass(frozen=True)
class GoldsetCsvApplyLabelsResult:
    input_path: str
    labels_path: str
    output_path: str
    summary_path: str | None
    summary: dict[str, Any]


@dataclass(frozen=True)
class GoldsetCsvLabelsTemplateResult:
    input_path: str
    output_path: str
    summary_path: str | None
    summary: dict[str, Any]


@dataclass(frozen=True)
class GoldsetCsvLabelsFromCsvResult:
    input_path: str
    output_path: str
    summary_path: str | None
    summary: dict[str, Any]


@dataclass(frozen=True)
class GoldsetCsvValidateLabelsResult:
    input_path: str
    summary_path: str | None
    summary: dict[str, Any]


@dataclass(frozen=True)
class GoldsetCsvSuggestLabelsResult:
    input_path: str
    output_path: str
    summary_path: str | None
    summary: dict[str, Any]


@dataclass(frozen=True)
class GoldsetCsvMergeSuggestionsResult:
    input_path: str
    suggestions_path: str
    output_path: str
    summary_path: str | None
    summary: dict[str, Any]


@dataclass(frozen=True)
class GoldsetCsvReviewPlanResult:
    input_path: str
    output_path: str
    summary_path: str | None
    summary: dict[str, Any]


def export_goldset_template_csv(
    *,
    input_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
) -> GoldsetCsvResult:
    source_path = Path(input_path)
    rows = tuple(_load_jsonl(source_path))
    csv_rows = tuple(_csv_row(row) for row in rows)
    target = Path(output_path)
    _write_csv(target, csv_rows)
    summary = _export_summary(input_path=str(source_path), output_path=str(target), rows=csv_rows)
    summary_target = _write_json(summary_path, summary)
    return GoldsetCsvResult(
        input_path=str(source_path),
        output_path=str(target),
        summary_path=str(summary_target) if summary_target else None,
        summary=summary,
    )


def inspect_goldset_annotation_progress(
    *,
    input_path: str | Path,
    summary_path: str | Path | None = None,
    guide_path: str | Path | None = None,
    worklist_path: str | Path | None = None,
    min_samples: int = 20,
    min_label_count: int = 1,
) -> GoldsetCsvProgressResult:
    source_path = Path(input_path)
    rows = tuple(_load_csv(source_path))
    guide_target = Path(guide_path) if guide_path is not None else None
    worklist_target = Path(worklist_path) if worklist_path is not None else None
    worklist_rows = tuple(_annotation_worklist_rows(rows))
    summary = _progress_summary(
        input_path=str(source_path),
        guide_path=str(guide_target) if guide_target is not None else None,
        worklist_path=str(worklist_target) if worklist_target is not None else None,
        rows=rows,
        worklist_rows=worklist_rows,
        min_samples=max(0, min_samples),
        min_label_count=max(0, min_label_count),
    )
    summary_target = _write_json(summary_path, summary)
    if guide_target is not None:
        _write_text(guide_target, _render_annotation_guide(summary))
    if worklist_target is not None:
        _write_jsonl(worklist_target, worklist_rows)
    return GoldsetCsvProgressResult(
        input_path=str(source_path),
        summary_path=str(summary_target) if summary_target else None,
        guide_path=str(guide_target) if guide_target is not None else None,
        worklist_path=str(worklist_target) if worklist_target is not None else None,
        summary=summary,
    )


def export_goldset_labels_template(
    *,
    input_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
) -> GoldsetCsvLabelsTemplateResult:
    source_path = Path(input_path)
    rows = tuple(_load_jsonl(source_path))
    label_rows = tuple(_label_template_row(row, index=index) for index, row in enumerate(rows))
    target = Path(output_path)
    _write_jsonl(target, label_rows)
    summary = _labels_template_summary(
        input_path=str(source_path),
        output_path=str(target),
        rows=label_rows,
    )
    summary_target = _write_json(summary_path, summary)
    return GoldsetCsvLabelsTemplateResult(
        input_path=str(source_path),
        output_path=str(target),
        summary_path=str(summary_target) if summary_target else None,
        summary=summary,
    )


def export_goldset_labels_from_csv(
    *,
    input_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    min_samples: int = 20,
    min_label_count: int = 1,
) -> GoldsetCsvLabelsFromCsvResult:
    source_path = Path(input_path)
    rows = tuple(_load_csv(source_path))
    label_rows = tuple(_label_row_from_csv(row, index=index) for index, row in enumerate(rows))
    target = Path(output_path)
    _write_jsonl(target, label_rows)
    summary = _labels_from_csv_summary(
        input_path=str(source_path),
        output_path=str(target),
        rows=label_rows,
        source_rows=rows,
        min_samples=max(0, min_samples),
        min_label_count=max(0, min_label_count),
    )
    summary_target = _write_json(summary_path, summary)
    return GoldsetCsvLabelsFromCsvResult(
        input_path=str(source_path),
        output_path=str(target),
        summary_path=str(summary_target) if summary_target else None,
        summary=summary,
    )


def validate_goldset_labels_jsonl(
    *,
    input_path: str | Path,
    summary_path: str | Path | None = None,
    require_complete: bool = False,
    min_samples: int = 20,
    min_label_count: int = 1,
) -> GoldsetCsvValidateLabelsResult:
    source_path = Path(input_path)
    rows = tuple(_load_jsonl(source_path))
    summary = _validate_labels_summary(
        input_path=str(source_path),
        rows=rows,
        require_complete=require_complete,
        min_samples=max(0, min_samples),
        min_label_count=max(0, min_label_count),
    )
    summary_target = _write_json(summary_path, summary)
    return GoldsetCsvValidateLabelsResult(
        input_path=str(source_path),
        summary_path=str(summary_target) if summary_target else None,
        summary=summary,
    )


def suggest_goldset_annotation_labels(
    *,
    input_path: str | Path,
    output_path: str | Path,
    base_url: str,
    model: str,
    api_key: str | None = None,
    timeout: float = 30.0,
    min_confidence: float = 0.66,
    max_rows: int | None = None,
    summary_path: str | Path | None = None,
) -> GoldsetCsvSuggestLabelsResult:
    source_path = Path(input_path)
    target = Path(output_path)
    rows = tuple(_load_csv(source_path))
    normalized_max_rows = _normalize_optional_nonnegative_int(max_rows)
    selected_rows = rows[:normalized_max_rows] if normalized_max_rows is not None else rows
    config = LocalModelJudgeConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout,
    )
    suggestions = tuple(
        _suggestion_row(
            row,
            index=index,
            config=config,
            min_confidence=min_confidence,
        )
        for index, row in enumerate(selected_rows)
    )
    _write_jsonl(target, suggestions)
    summary = _suggestions_summary(
        input_path=str(source_path),
        output_path=str(target),
        base_url=base_url,
        model=model,
        api_key_configured=bool(api_key),
        timeout=timeout,
        min_confidence=min_confidence,
        max_rows=normalized_max_rows,
        source_row_count=len(rows),
        suggestions=suggestions,
    )
    summary_target = _write_json(summary_path, summary)
    return GoldsetCsvSuggestLabelsResult(
        input_path=str(source_path),
        output_path=str(target),
        summary_path=str(summary_target) if summary_target else None,
        summary=summary,
    )


def merge_goldset_annotation_suggestions(
    *,
    input_path: str | Path,
    suggestions_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
) -> GoldsetCsvMergeSuggestionsResult:
    source_path = Path(input_path)
    suggestion_path = Path(suggestions_path)
    target = Path(output_path)
    rows = tuple(_load_csv(source_path))
    suggestions = tuple(_load_jsonl(suggestion_path))
    merged_rows, diagnostics = _merge_suggestion_rows(rows=rows, suggestions=suggestions)
    _write_suggestion_review_csv(target, merged_rows)
    summary = _merge_suggestions_summary(
        input_path=str(source_path),
        suggestions_path=str(suggestion_path),
        output_path=str(target),
        rows=merged_rows,
        diagnostics=diagnostics,
    )
    summary_target = _write_json(summary_path, summary)
    return GoldsetCsvMergeSuggestionsResult(
        input_path=str(source_path),
        suggestions_path=str(suggestion_path),
        output_path=str(target),
        summary_path=str(summary_target) if summary_target else None,
        summary=summary,
    )


def build_goldset_annotation_review_plan(
    *,
    input_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    max_rows: int | None = None,
) -> GoldsetCsvReviewPlanResult:
    source_path = Path(input_path)
    target = Path(output_path)
    rows = tuple(_load_csv(source_path))
    plan_rows = tuple(_review_plan_rows(rows, max_rows=_normalize_optional_nonnegative_int(max_rows)))
    _write_jsonl(target, plan_rows)
    summary = _review_plan_summary(
        input_path=str(source_path),
        output_path=str(target),
        rows=rows,
        plan_rows=plan_rows,
        max_rows=_normalize_optional_nonnegative_int(max_rows),
    )
    summary_target = _write_json(summary_path, summary)
    return GoldsetCsvReviewPlanResult(
        input_path=str(source_path),
        output_path=str(target),
        summary_path=str(summary_target) if summary_target else None,
        summary=summary,
    )


def apply_goldset_annotation_labels(
    *,
    input_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    require_complete: bool = False,
) -> GoldsetCsvApplyLabelsResult:
    source_path = Path(input_path)
    label_path = Path(labels_path)
    target = Path(output_path)
    rows = tuple(_load_csv(source_path))
    label_rows = tuple(_load_jsonl(label_path))
    updated_rows, diagnostics = _apply_label_rows(rows=rows, label_rows=label_rows)
    summary = _apply_labels_summary(
        input_path=str(source_path),
        labels_path=str(label_path),
        output_path=str(target),
        rows=updated_rows,
        diagnostics=diagnostics,
        require_complete=require_complete,
    )
    if summary["output_written"] is True:
        _write_csv(target, updated_rows)
    summary_target = _write_json(summary_path, summary)
    return GoldsetCsvApplyLabelsResult(
        input_path=str(source_path),
        labels_path=str(label_path),
        output_path=str(target),
        summary_path=str(summary_target) if summary_target else None,
        summary=summary,
    )


def import_goldset_labeled_csv(
    *,
    input_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    allow_incomplete_output: bool = False,
) -> GoldsetCsvResult:
    source_path = Path(input_path)
    rows = tuple(_load_csv(source_path))
    jsonl_rows = tuple(_jsonl_row(row, index=index) for index, row in enumerate(rows))
    target = Path(output_path)
    summary = _import_summary(
        input_path=str(source_path),
        output_path=str(target),
        rows=jsonl_rows,
        allow_incomplete_output=allow_incomplete_output,
    )
    if summary["ready_for_local_judge_goldset_validate"] is True or allow_incomplete_output:
        _write_jsonl(target, jsonl_rows)
    summary_target = _write_json(summary_path, summary)
    return GoldsetCsvResult(
        input_path=str(source_path),
        output_path=str(target),
        summary_path=str(summary_target) if summary_target else None,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export/import local judge gold-set templates as CSV for manual annotation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export a with-text gold-set JSONL template to CSV.")
    export_parser.add_argument("--input", required=True, help="Input local gold-set template JSONL.")
    export_parser.add_argument("--output", required=True, help="Output local annotation CSV.")
    export_parser.add_argument("--summary", help="Optional prompt-safe export summary JSON.")

    progress_parser = subparsers.add_parser("progress", help="Inspect prompt-safe manual annotation progress.")
    progress_parser.add_argument("--input", required=True, help="Input local annotation CSV.")
    progress_parser.add_argument("--summary", help="Optional prompt-safe progress summary JSON.")
    progress_parser.add_argument("--guide-md", help="Optional prompt-safe Markdown annotation guide.")
    progress_parser.add_argument("--worklist-jsonl", help="Optional prompt-safe JSONL worklist of rows needing labels.")
    progress_parser.add_argument("--min-samples", type=int, default=20)
    progress_parser.add_argument("--min-label-count", type=int, default=1)

    apply_parser = subparsers.add_parser(
        "apply-labels",
        help="Apply prompt-safe JSONL expected_label updates to a local annotation CSV.",
    )
    apply_parser.add_argument("--input", required=True, help="Input local annotation CSV containing candidate text.")
    apply_parser.add_argument(
        "--labels",
        required=True,
        help="Prompt-safe JSONL rows with id, text_hash, and expected_label fields.",
    )
    apply_parser.add_argument("--output", required=True, help="Output updated local labeled annotation CSV.")
    apply_parser.add_argument("--summary", help="Optional prompt-safe label-apply summary JSON.")
    apply_parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Write output only if every CSV row has a complete, valid expected_label after applying labels.",
    )

    labels_template_parser = subparsers.add_parser(
        "labels-template",
        help="Create a prompt-safe labels JSONL template from an annotation worklist.",
    )
    labels_template_parser.add_argument("--input", required=True, help="Input prompt-safe annotation worklist JSONL.")
    labels_template_parser.add_argument(
        "--output",
        required=True,
        help="Output prompt-safe labels JSONL template to be filled with expected_label values.",
    )
    labels_template_parser.add_argument("--summary", help="Optional prompt-safe labels-template summary JSON.")

    labels_from_csv_parser = subparsers.add_parser(
        "labels-from-csv",
        help="Export prompt-safe expected_label JSONL from a local reviewed annotation CSV.",
    )
    labels_from_csv_parser.add_argument(
        "--input",
        required=True,
        help="Input local annotation or suggestion-review CSV containing manually filled expected_label values.",
    )
    labels_from_csv_parser.add_argument(
        "--output",
        required=True,
        help="Output prompt-safe labels JSONL for validate-labels/apply-labels.",
    )
    labels_from_csv_parser.add_argument("--summary", help="Optional prompt-safe labels-from-csv summary JSON.")
    labels_from_csv_parser.add_argument("--min-samples", type=int, default=20)
    labels_from_csv_parser.add_argument("--min-label-count", type=int, default=1)

    validate_labels_parser = subparsers.add_parser(
        "validate-labels",
        help="Validate a prompt-safe labels JSONL before applying it to a local annotation CSV.",
    )
    validate_labels_parser.add_argument("--input", required=True, help="Input prompt-safe labels JSONL.")
    validate_labels_parser.add_argument("--summary", help="Optional prompt-safe labels validation summary JSON.")
    validate_labels_parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Return success only if every labels row has a valid expected_label.",
    )
    validate_labels_parser.add_argument("--min-samples", type=int, default=20)
    validate_labels_parser.add_argument("--min-label-count", type=int, default=1)

    suggest_labels_parser = subparsers.add_parser(
        "suggest-labels",
        help="Call a local judge to write prompt-safe label suggestions for a local annotation CSV.",
    )
    suggest_labels_parser.add_argument("--input", required=True, help="Input local annotation CSV with candidate text.")
    suggest_labels_parser.add_argument(
        "--output",
        required=True,
        help="Output prompt-safe suggestions JSONL. This is not an expected_label file.",
    )
    suggest_labels_parser.add_argument("--base-url", required=True, help="OpenAI-compatible local judge base URL.")
    suggest_labels_parser.add_argument("--model", required=True, help="Local judge model name.")
    suggest_labels_parser.add_argument("--api-key-env", help="Optional environment variable for local judge API key.")
    suggest_labels_parser.add_argument("--timeout", type=float, default=30.0)
    suggest_labels_parser.add_argument("--min-confidence", type=float, default=0.66)
    suggest_labels_parser.add_argument("--max-rows", type=int)
    suggest_labels_parser.add_argument("--summary", help="Optional prompt-safe suggestions summary JSON.")

    merge_suggestions_parser = subparsers.add_parser(
        "merge-suggestions",
        help="Create a local CSV for manual review by joining annotation text with prompt-safe suggestions.",
    )
    merge_suggestions_parser.add_argument("--input", required=True, help="Input local annotation CSV with candidate text.")
    merge_suggestions_parser.add_argument(
        "--suggestions",
        required=True,
        help="Prompt-safe suggestions JSONL from suggest-labels.",
    )
    merge_suggestions_parser.add_argument(
        "--output",
        required=True,
        help="Output local manual-review CSV. This contains candidate text and must stay local.",
    )
    merge_suggestions_parser.add_argument("--summary", help="Optional prompt-safe merge summary JSON.")

    review_plan_parser = subparsers.add_parser(
        "review-plan",
        help="Create a prompt-safe priority worklist from a local suggestion-review CSV.",
    )
    review_plan_parser.add_argument(
        "--input",
        required=True,
        help="Input local suggestion-review CSV. It may contain candidate text, but output omits it.",
    )
    review_plan_parser.add_argument(
        "--output",
        required=True,
        help="Output prompt-safe JSONL review plan for manual expected_label work.",
    )
    review_plan_parser.add_argument("--summary", help="Optional prompt-safe review-plan summary JSON.")
    review_plan_parser.add_argument("--max-rows", type=int)

    import_parser = subparsers.add_parser("import", help="Import a labeled CSV into local gold JSONL.")
    import_parser.add_argument("--input", required=True, help="Input local labeled annotation CSV.")
    import_parser.add_argument("--output", required=True, help="Output local labeled gold JSONL.")
    import_parser.add_argument("--summary", help="Optional prompt-safe import summary JSON.")
    import_parser.add_argument(
        "--allow-incomplete-output",
        action="store_true",
        help="Write output JSONL even when labels/text are incomplete. Default is fail-fast without writing.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export":
        result = export_goldset_template_csv(
            input_path=args.input,
            output_path=args.output,
            summary_path=args.summary,
        )
    elif args.command == "progress":
        result = inspect_goldset_annotation_progress(
            input_path=args.input,
            summary_path=args.summary,
            guide_path=args.guide_md,
            worklist_path=args.worklist_jsonl,
            min_samples=args.min_samples,
            min_label_count=args.min_label_count,
        )
    elif args.command == "apply-labels":
        result = apply_goldset_annotation_labels(
            input_path=args.input,
            labels_path=args.labels,
            output_path=args.output,
            summary_path=args.summary,
            require_complete=args.require_complete,
        )
    elif args.command == "labels-template":
        result = export_goldset_labels_template(
            input_path=args.input,
            output_path=args.output,
            summary_path=args.summary,
        )
    elif args.command == "labels-from-csv":
        result = export_goldset_labels_from_csv(
            input_path=args.input,
            output_path=args.output,
            summary_path=args.summary,
            min_samples=args.min_samples,
            min_label_count=args.min_label_count,
        )
    elif args.command == "validate-labels":
        result = validate_goldset_labels_jsonl(
            input_path=args.input,
            summary_path=args.summary,
            require_complete=args.require_complete,
            min_samples=args.min_samples,
            min_label_count=args.min_label_count,
        )
    elif args.command == "suggest-labels":
        import os

        result = suggest_goldset_annotation_labels(
            input_path=args.input,
            output_path=args.output,
            base_url=args.base_url,
            model=args.model,
            api_key=os.environ.get(args.api_key_env) if args.api_key_env else None,
            timeout=args.timeout,
            min_confidence=args.min_confidence,
            max_rows=args.max_rows,
            summary_path=args.summary,
        )
    elif args.command == "merge-suggestions":
        result = merge_goldset_annotation_suggestions(
            input_path=args.input,
            suggestions_path=args.suggestions,
            output_path=args.output,
            summary_path=args.summary,
        )
    elif args.command == "review-plan":
        result = build_goldset_annotation_review_plan(
            input_path=args.input,
            output_path=args.output,
            summary_path=args.summary,
            max_rows=args.max_rows,
        )
    elif args.command == "import":
        result = import_goldset_labeled_csv(
            input_path=args.input,
            output_path=args.output,
            summary_path=args.summary,
            allow_incomplete_output=args.allow_incomplete_output,
        )
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    if args.command == "apply-labels":
        return 0 if result.summary.get("output_written") is True else 1
    if args.command == "validate-labels":
        return 0 if result.summary.get("ready_for_apply_labels") is True else 1
    if args.command == "labels-from-csv":
        return 0 if result.summary.get("ready_for_apply_labels") is True else 1
    if args.command == "suggest-labels":
        return 0 if result.summary.get("suggestions_ready_for_manual_review") is True else 1
    if args.command == "merge-suggestions":
        return 0 if result.summary.get("ready_for_manual_review") is True else 1
    if args.command == "review-plan":
        return (
            0
            if (
                result.summary.get("ready_for_manual_review") is True
                or result.summary.get("annotation_complete") is True
            )
            else 1
        )
    if args.command == "import":
        return 0 if result.summary.get("ready_for_local_judge_goldset_validate") is True else 1
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


def _load_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {str(key): value or "" for key, value in row.items() if key is not None}


def _csv_row(row: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for column in CSV_COLUMNS:
        value = row.get(column)
        if column in {"label_options", "risk_tags"}:
            result[column] = json.dumps(value or [], ensure_ascii=False)
        elif column == "expected_label":
            result[column] = str(value or "").strip().lower()
        else:
            result[column] = "" if value is None else str(value)
    return result


def _jsonl_row(row: Mapping[str, str], *, index: int) -> dict[str, Any]:
    text = row.get("text", "")
    expected = str(row.get("expected_label") or "").strip().lower()
    item: dict[str, Any] = {
        "schema_version": "prefix-local-judge-goldset-labeled-item-v1",
        "id": str(row.get("id") or f"gold-row-{index}").strip() or f"gold-row-{index}",
        "expected_label": expected if expected else None,
        "annotation_notes": row.get("annotation_notes") or None,
        "label_options": _parse_list(row.get("label_options")) or sorted(ALLOWED_LABELS),
        "semantic_hint": row.get("semantic_hint") or None,
        "risk_tags": _parse_list(row.get("risk_tags")),
        "source_label": row.get("source_label") or None,
        "rule_label": row.get("rule_label") or None,
        "model_label": row.get("model_label") or None,
        "review_priority": row.get("review_priority") or None,
        "recommended_action": row.get("recommended_action") or None,
        "source_path": row.get("source_path") or None,
        "symbol": row.get("symbol") or None,
        "extraction": row.get("extraction") or None,
        "parent_hash": row.get("parent_hash") or None,
        "text_hash": row.get("text_hash") or _text_hash(text),
        "text": text,
    }
    return {key: value for key, value in item.items() if value is not None}


def _parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in text.split(";")]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    return [str(parsed)] if str(parsed) else []


def _export_summary(*, input_path: str, output_path: str, rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    expected_counts = Counter(_valid_label(row.get("expected_label")) for row in rows)
    expected_counts.pop("", None)
    rows_with_text = sum(1 for row in rows if str(row.get("text") or "").strip())
    rows_with_labels = sum(1 for row in rows if _valid_label(row.get("expected_label")))
    return {
        "schema_version": "prefix-local-judge-goldset-csv-export-summary-v1",
        "prompt_safe_summary": True,
        "input_path": input_path,
        "output_path": output_path,
        "csv_text_written": rows_with_text > 0,
        "gold_text_written": False,
        "row_count": len(rows),
        "rows_with_text_count": rows_with_text,
        "rows_missing_text_count": len(rows) - rows_with_text,
        "rows_with_expected_label_count": rows_with_labels,
        "expected_label_pending_count": len(rows) - rows_with_labels,
        "expected_label_counts": dict(sorted(expected_counts.items())),
        "label_options": sorted(ALLOWED_LABELS),
        "text_artifact_note": "output CSV contains raw candidate text; keep it local and do not commit",
        "ready_for_local_judge_quality_eval": False,
        "recommendation": "fill_expected_label_in_csv_then_import_labeled_jsonl",
        "limits": (
            "This is a local annotation CSV export. It does not call a local judge, provider API, "
            "or prove semantic quality, cache, latency, cost, or task success."
        ),
    }


def _progress_summary(
    *,
    input_path: str,
    guide_path: str | None,
    worklist_path: str | None,
    rows: Sequence[Mapping[str, str]],
    worklist_rows: Sequence[Mapping[str, Any]],
    min_samples: int,
    min_label_count: int,
) -> dict[str, Any]:
    row_count = len(rows)
    rows_with_text = sum(1 for row in rows if str(row.get("text") or "").strip())
    rows_missing_text = row_count - rows_with_text
    valid_expected_labels = [_valid_label(row.get("expected_label")) for row in rows]
    expected_counts = Counter(label for label in valid_expected_labels if label)
    rows_with_labels = sum(1 for label in valid_expected_labels if label)
    pending_label_count = sum(1 for row in rows if not str(row.get("expected_label") or "").strip())
    invalid_label_count = sum(
        1
        for row in rows
        if str(row.get("expected_label") or "").strip() and not _valid_label(row.get("expected_label"))
    )
    pending_rows = tuple(row for row in rows if not str(row.get("expected_label") or "").strip())
    invalid_rows = tuple(
        row
        for row in rows
        if str(row.get("expected_label") or "").strip() and not _valid_label(row.get("expected_label"))
    )
    labels_with_min = sum(1 for count in expected_counts.values() if count >= min_label_count)
    ready_for_import = bool(row_count > 0 and rows_missing_text == 0 and pending_label_count == 0 and invalid_label_count == 0)
    ready_after_import = bool(
        ready_for_import
        and row_count >= min_samples
        and labels_with_min >= len(ALLOWED_LABELS)
    )
    return {
        "schema_version": "prefix-local-judge-goldset-annotation-progress-summary-v1",
        "prompt_safe_summary": True,
        "input_path": input_path,
        "guide_path": guide_path,
        "worklist_path": worklist_path,
        "row_count": row_count,
        "rows_with_text_count": rows_with_text,
        "rows_missing_text_count": rows_missing_text,
        "rows_with_expected_label_count": rows_with_labels,
        "expected_label_pending_count": pending_label_count,
        "invalid_expected_label_count": invalid_label_count,
        "expected_label_counts": dict(sorted(expected_counts.items())),
        "annotation_completion_rate": (rows_with_labels / row_count) if row_count else 0.0,
        "min_samples": min_samples,
        "min_label_count": min_label_count,
        "labels_with_min_count": labels_with_min,
        "required_label_count": len(ALLOWED_LABELS),
        "label_balance_ready": labels_with_min >= len(ALLOWED_LABELS),
        "label_options": sorted(ALLOWED_LABELS),
        "source_label_counts": _value_counts(rows, "source_label"),
        "semantic_hint_counts": _value_counts(rows, "semantic_hint"),
        "review_priority_counts": _value_counts(rows, "review_priority"),
        "risk_tag_counts": _list_counts(rows, "risk_tags"),
        "pending_source_label_counts": _value_counts(pending_rows, "source_label"),
        "pending_semantic_hint_counts": _value_counts(pending_rows, "semantic_hint"),
        "pending_review_priority_counts": _value_counts(pending_rows, "review_priority"),
        "pending_risk_tag_counts": _list_counts(pending_rows, "risk_tags"),
        "invalid_source_label_counts": _value_counts(invalid_rows, "source_label"),
        "invalid_semantic_hint_counts": _value_counts(invalid_rows, "semantic_hint"),
        "invalid_review_priority_counts": _value_counts(invalid_rows, "review_priority"),
        "invalid_risk_tag_counts": _list_counts(invalid_rows, "risk_tags"),
        "worklist_written": worklist_path is not None,
        "worklist_row_count": len(worklist_rows),
        "worklist_pending_count": sum(1 for row in worklist_rows if row.get("label_status") == "pending"),
        "worklist_invalid_count": sum(1 for row in worklist_rows if row.get("label_status") == "invalid"),
        "worklist_schema_version": "prefix-local-judge-goldset-annotation-worklist-item-v1",
        "ready_for_labeled_jsonl_import": ready_for_import,
        "ready_for_local_judge_goldset_validate_after_import": ready_after_import,
        "ready_for_local_judge_quality_eval_after_import": ready_after_import,
        "text_artifact_note": "input CSV contains raw candidate text; keep it local and do not commit",
        "recommendation": _progress_recommendation(
            ready_after_import=ready_after_import,
            ready_for_import=ready_for_import,
            row_count=row_count,
            pending_label_count=pending_label_count,
            invalid_label_count=invalid_label_count,
            rows_missing_text=rows_missing_text,
            labels_with_min_count=labels_with_min,
            min_samples=min_samples,
        ),
        "limits": (
            "This inspects local human annotation progress only. It does not write labeled JSONL, call a local judge, "
            "call a provider API, or prove semantic quality, cache, latency, cost, or task success."
        ),
    }


def _progress_recommendation(
    *,
    ready_after_import: bool,
    ready_for_import: bool,
    row_count: int,
    pending_label_count: int,
    invalid_label_count: int,
    rows_missing_text: int,
    labels_with_min_count: int,
    min_samples: int,
) -> str:
    if ready_after_import:
        return "import_labeled_csv_then_validate_goldset"
    if row_count == 0:
        return "build_annotation_csv_before_labeling"
    if rows_missing_text > 0:
        return "restore_candidate_text_before_labeling"
    if pending_label_count > 0:
        return "fill_expected_label_for_pending_rows"
    if invalid_label_count > 0:
        return "fix_invalid_expected_label_values"
    if row_count < min_samples:
        return "add_more_gold_labeled_rows"
    if labels_with_min_count < len(ALLOWED_LABELS):
        return "add_label_balance_before_quality_eval"
    if ready_for_import:
        return "import_labeled_csv_then_validate_goldset"
    return "inspect_annotation_progress_summary"


def _apply_label_rows(
    *,
    rows: Sequence[Mapping[str, str]],
    label_rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, str], ...], dict[str, Any]]:
    updated_rows = tuple(dict(row) for row in rows)
    row_key_to_index: dict[tuple[str, str], int] = {}
    duplicate_csv_key_count = 0
    for index, row in enumerate(updated_rows):
        key = _csv_label_key(row, index=index)
        if key in row_key_to_index:
            duplicate_csv_key_count += 1
            continue
        row_key_to_index[key] = index

    updates: dict[tuple[str, str], str] = {}
    label_expected_counts: Counter[str] = Counter()
    pending_label_row_count = 0
    invalid_label_row_count = 0
    missing_label_key_count = 0
    duplicate_label_key_count = 0
    unmatched_label_count = 0
    valid_label_row_count = 0

    for label_row in label_rows:
        key = _jsonl_label_key(label_row)
        raw_label = str(label_row.get("expected_label") or label_row.get("label") or "").strip()
        label = _valid_label(raw_label)
        if not raw_label:
            pending_label_row_count += 1
            continue
        if not label:
            invalid_label_row_count += 1
            continue
        if key is None:
            missing_label_key_count += 1
            continue
        if key in updates:
            duplicate_label_key_count += 1
            continue
        if key not in row_key_to_index:
            unmatched_label_count += 1
            continue
        valid_label_row_count += 1
        label_expected_counts[label] += 1
        updates[key] = label

    if (
        duplicate_csv_key_count == 0
        and pending_label_row_count == 0
        and invalid_label_row_count == 0
        and missing_label_key_count == 0
        and duplicate_label_key_count == 0
        and unmatched_label_count == 0
    ):
        for key, label in updates.items():
            updated_rows[row_key_to_index[key]]["expected_label"] = label

    diagnostics = {
        "label_row_count": len(label_rows),
        "valid_label_row_count": valid_label_row_count,
        "pending_label_row_count": pending_label_row_count,
        "invalid_label_row_count": invalid_label_row_count,
        "missing_label_key_count": missing_label_key_count,
        "duplicate_label_key_count": duplicate_label_key_count,
        "unmatched_label_count": unmatched_label_count,
        "duplicate_csv_key_count": duplicate_csv_key_count,
        "matched_label_count": len(updates),
        "label_expected_counts": dict(sorted(label_expected_counts.items())),
    }
    return updated_rows, diagnostics


def _label_template_row(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    item: dict[str, Any] = {
        "schema_version": "prefix-local-judge-goldset-annotation-label-v1",
        "id": str(row.get("id") or f"gold-row-{index}").strip() or f"gold-row-{index}",
        "text_hash": str(row.get("text_hash") or "").strip(),
        "expected_label": "",
        "annotation_notes": "",
        "label_options": _parse_list(row.get("label_options")) or sorted(ALLOWED_LABELS),
        "source_label": row.get("source_label") or None,
        "rule_label": row.get("rule_label") or None,
        "semantic_hint": row.get("semantic_hint") or None,
        "risk_tags": _parse_list(row.get("risk_tags")),
        "label_status": row.get("label_status") or None,
        "recommended_action": row.get("recommended_action") or "fill_expected_label",
        "csv_row_number": row.get("csv_row_number") or None,
    }
    return {key: value for key, value in item.items() if value not in (None, [])}


def _label_row_from_csv(row: Mapping[str, str], *, index: int) -> dict[str, Any]:
    expected_label = str(row.get("expected_label") or "").strip().lower()
    item: dict[str, Any] = {
        "schema_version": "prefix-local-judge-goldset-annotation-label-v1",
        "id": str(row.get("id") or f"gold-row-{index}").strip() or f"gold-row-{index}",
        "text_hash": str(row.get("text_hash") or _text_hash(row.get("text", ""))).strip(),
        "expected_label": expected_label,
        "annotation_notes": row.get("annotation_notes") or "",
        "label_options": _parse_list(row.get("label_options")) or sorted(ALLOWED_LABELS),
        "source_label": row.get("source_label") or None,
        "rule_label": row.get("rule_label") or None,
        "semantic_hint": row.get("semantic_hint") or None,
        "risk_tags": _parse_list(row.get("risk_tags")),
        "suggested_label": _valid_label(row.get("suggested_label")) or None,
        "suggestion_notes": row.get("suggestion_notes") or None,
        "recommended_action": "validate_labels_then_apply_to_annotation_csv",
        "csv_row_number": index + 2,
    }
    return {key: value for key, value in item.items() if value not in (None, [])}


def _review_plan_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    max_rows: int | None,
) -> Iterable[dict[str, Any]]:
    pending_rows = [
        _review_plan_row(row, index=index)
        for index, row in enumerate(rows)
        if not _valid_label(row.get("expected_label"))
    ]
    sorted_rows = sorted(pending_rows, key=_review_plan_sort_key)
    selected_rows = sorted_rows[:max_rows] if max_rows is not None else sorted_rows
    for priority, row in enumerate(selected_rows, start=1):
        yield {**row, "review_priority_rank": priority}


def _review_plan_row(row: Mapping[str, str], *, index: int) -> dict[str, Any]:
    source_label = _valid_label(row.get("source_label")) or None
    rule_label = _valid_label(row.get("rule_label")) or None
    model_label = _valid_label(row.get("model_label")) or None
    suggested_label = _valid_label(row.get("suggested_label")) or None
    expected_label = _valid_label(row.get("expected_label")) or ""
    raw_expected = str(row.get("expected_label") or "").strip()
    risk_tags = _parse_list(row.get("risk_tags"))
    label_values = [label for label in (source_label, rule_label, model_label, suggested_label) if label]
    distinct_labels = sorted(set(label_values))
    disagreement_count = max(0, len(distinct_labels) - 1)
    high_risk_tags = sorted(set(risk_tags) & HIGH_RISK_TAGS)
    reasons = _review_plan_reasons(
        raw_expected=raw_expected,
        expected_label=expected_label,
        disagreement_count=disagreement_count,
        high_risk_tags=high_risk_tags,
        source_label=source_label,
        rule_label=rule_label,
        model_label=model_label,
        suggested_label=suggested_label,
    )
    text = str(row.get("text") or "")
    item: dict[str, Any] = {
        "schema_version": "prefix-local-judge-goldset-review-plan-item-v1",
        "csv_row_number": index + 2,
        "id": str(row.get("id") or f"gold-row-{index}").strip() or f"gold-row-{index}",
        "text_hash": str(row.get("text_hash") or _text_hash(text)).strip(),
        "expected_label_status": "invalid" if raw_expected and not expected_label else "pending",
        "expected_label": raw_expected if raw_expected and not expected_label else None,
        "source_label": source_label,
        "rule_label": rule_label,
        "model_label": model_label,
        "suggested_label": suggested_label,
        "model_confidence": _optional_float(row.get("model_confidence")),
        "confidence_meets_min": _optional_bool(row.get("confidence_meets_min")),
        "static_safety_clamped": _optional_bool(row.get("static_safety_clamped")),
        "local_judge_action": row.get("local_judge_action") or None,
        "label_reason_code": row.get("label_reason_code") or None,
        "semantic_hint": row.get("semantic_hint") or None,
        "risk_tags": risk_tags,
        "high_risk_tags": high_risk_tags,
        "label_disagreement_count": disagreement_count,
        "distinct_candidate_labels": distinct_labels,
        "review_reasons": reasons,
        "recommended_action": "read_candidate_text_and_fill_expected_label",
    }
    return {key: value for key, value in item.items() if value not in (None, "", [])}


def _review_plan_reasons(
    *,
    raw_expected: str,
    expected_label: str,
    disagreement_count: int,
    high_risk_tags: Sequence[str],
    source_label: str | None,
    rule_label: str | None,
    model_label: str | None,
    suggested_label: str | None,
) -> list[str]:
    reasons: list[str] = []
    if raw_expected and not expected_label:
        reasons.append("invalid_expected_label")
    else:
        reasons.append("missing_expected_label")
    if disagreement_count > 0:
        reasons.append("source_rule_model_or_suggestion_disagree")
    if high_risk_tags:
        reasons.append("high_risk_boundary")
    if source_label == "review" or rule_label == "review" or suggested_label == "review":
        reasons.append("review_label_present")
    if model_label and suggested_label and model_label != suggested_label:
        reasons.append("model_label_clamped_or_adjusted")
    if suggested_label and suggested_label != source_label:
        reasons.append("suggestion_differs_from_source_label")
    return reasons


def _review_plan_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, int, int, int, str]:
    reasons = set(_parse_list(row.get("review_reasons")))
    labels = set(_parse_list(row.get("distinct_candidate_labels")))
    return (
        0 if row.get("expected_label_status") == "invalid" else 1,
        -_num(row.get("label_disagreement_count")),
        -len(_parse_list(row.get("high_risk_tags"))),
        0 if "review" in labels else 1,
        0 if "source_rule_model_or_suggestion_disagree" in reasons else 1,
        _num(row.get("csv_row_number")),
        str(row.get("id") or ""),
    )


def _labels_template_summary(
    *,
    input_path: str,
    output_path: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows_missing_id = sum(1 for row in rows if not str(row.get("id") or "").strip())
    rows_missing_text_hash = sum(1 for row in rows if not str(row.get("text_hash") or "").strip())
    row_keys = [_jsonl_label_key(row) for row in rows]
    key_counts = Counter(key for key in row_keys if key is not None)
    duplicate_key_count = sum(1 for count in key_counts.values() if count > 1)
    return {
        "schema_version": "prefix-local-judge-goldset-labels-template-summary-v1",
        "prompt_safe_summary": True,
        "input_path": input_path,
        "output_path": output_path,
        "output_written": True,
        "row_count": len(rows),
        "rows_missing_id_count": rows_missing_id,
        "rows_missing_text_hash_count": rows_missing_text_hash,
        "duplicate_label_key_count": duplicate_key_count,
        "expected_label_pending_count": len(rows),
        "expected_label_counts": {},
        "source_label_counts": _any_value_counts(rows, "source_label"),
        "semantic_hint_counts": _any_value_counts(rows, "semantic_hint"),
        "risk_tag_counts": _any_list_counts(rows, "risk_tags"),
        "label_status_counts": _any_value_counts(rows, "label_status"),
        "label_options": sorted(ALLOWED_LABELS),
        "ready_for_apply_labels": bool(
            len(rows) > 0
            and rows_missing_id == 0
            and rows_missing_text_hash == 0
            and duplicate_key_count == 0
        ),
        "text_artifact_note": "labels template is prompt-safe and omits candidate text, source_path, and symbol",
        "recommendation": "fill_expected_label_values_then_run_apply_labels",
        "limits": (
            "This writes a prompt-safe local labels template only. It does not apply labels, call a local judge, "
            "call a provider API, or prove semantic quality, cache, latency, cost, or task success."
        ),
    }


def _review_plan_summary(
    *,
    input_path: str,
    output_path: str,
    rows: Sequence[Mapping[str, str]],
    plan_rows: Sequence[Mapping[str, Any]],
    max_rows: int | None,
) -> dict[str, Any]:
    pending_count = sum(1 for row in rows if not str(row.get("expected_label") or "").strip())
    invalid_count = sum(
        1
        for row in rows
        if str(row.get("expected_label") or "").strip() and not _valid_label(row.get("expected_label"))
    )
    disagreement_rows = [
        row for row in plan_rows if _num(row.get("label_disagreement_count")) > 0
    ]
    high_risk_rows = [row for row in plan_rows if _parse_list(row.get("high_risk_tags"))]
    annotation_complete = bool(len(rows) > 0 and pending_count == 0 and invalid_count == 0)
    ready = bool(len(rows) > 0 and len(plan_rows) > 0)
    return {
        "schema_version": "prefix-local-judge-goldset-review-plan-summary-v1",
        "prompt_safe_summary": True,
        "input_path": input_path,
        "output_path": output_path,
        "output_written": True,
        "max_rows": max_rows,
        "row_count": len(rows),
        "plan_row_count": len(plan_rows),
        "expected_label_pending_count": pending_count,
        "invalid_expected_label_count": invalid_count,
        "source_label_counts": _value_counts(rows, "source_label"),
        "rule_label_counts": _value_counts(rows, "rule_label"),
        "model_label_counts": _value_counts(rows, "model_label"),
        "suggested_label_counts": _value_counts(rows, "suggested_label"),
        "plan_expected_label_status_counts": _any_value_counts(plan_rows, "expected_label_status"),
        "plan_review_reason_counts": _any_list_counts(plan_rows, "review_reasons"),
        "plan_high_risk_tag_counts": _any_list_counts(plan_rows, "high_risk_tags"),
        "plan_semantic_hint_counts": _any_value_counts(plan_rows, "semantic_hint"),
        "plan_disagreement_row_count": len(disagreement_rows),
        "plan_high_risk_row_count": len(high_risk_rows),
        "ready_for_manual_review": ready,
        "annotation_complete": annotation_complete,
        "ready_for_apply_labels": False,
        "ready_for_goldset_import_after_apply": False,
        "suggested_labels_not_auto_applied": True,
        "text_artifact_note": "input CSV may contain raw candidate text; output review plan omits candidate text, source_path, and symbol",
        "recommendation": _review_plan_recommendation(
            ready_for_manual_review=ready,
            annotation_complete=annotation_complete,
        ),
        "limits": (
            "This writes a prompt-safe priority plan for manual labeling only. It does not auto-fill "
            "expected_label, does not use suggested_label as a gold label, does not call a local judge or "
            "provider API, and does not prove semantic quality, cache, latency, cost, or task success."
        ),
    }


def _review_plan_recommendation(
    *,
    ready_for_manual_review: bool,
    annotation_complete: bool,
) -> str:
    if annotation_complete:
        return "review_plan_complete_run_labels_from_csv"
    if ready_for_manual_review:
        return "manually_fill_expected_label_using_prioritized_review_plan"
    return "create_suggestion_review_csv_before_review_plan"


def _labels_from_csv_summary(
    *,
    input_path: str,
    output_path: str,
    rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, str]],
    min_samples: int,
    min_label_count: int,
) -> dict[str, Any]:
    validation = _validate_labels_summary(
        input_path=output_path,
        rows=rows,
        require_complete=True,
        min_samples=min_samples,
        min_label_count=min_label_count,
    )
    suggested_counts = Counter(_valid_label(row.get("suggested_label")) for row in source_rows)
    suggested_counts.pop("", None)
    expected_to_suggested = _transition_counts(rows, "expected_label", "suggested_label")
    return {
        "schema_version": "prefix-local-judge-goldset-labels-from-csv-summary-v1",
        "prompt_safe_summary": True,
        "input_path": input_path,
        "output_path": output_path,
        "output_written": True,
        "row_count": len(rows),
        "valid_label_row_count": validation["valid_label_row_count"],
        "expected_label_pending_count": validation["expected_label_pending_count"],
        "invalid_label_row_count": validation["invalid_label_row_count"],
        "rows_missing_id_count": validation["rows_missing_id_count"],
        "rows_missing_text_hash_count": validation["rows_missing_text_hash_count"],
        "duplicate_label_key_count": validation["duplicate_label_key_count"],
        "expected_label_counts": validation["expected_label_counts"],
        "min_samples": min_samples,
        "min_label_count": min_label_count,
        "sample_count_ready": validation["sample_count_ready"],
        "label_balance_ready": validation["label_balance_ready"],
        "suggested_label_counts": dict(sorted(suggested_counts.items())),
        "expected_to_suggested_label_counts": expected_to_suggested,
        "ready_for_apply_labels": validation["ready_for_apply_labels"],
        "ready_for_goldset_import_after_apply": validation["ready_for_goldset_import_after_apply"],
        "csv_text_read": any(str(row.get("text") or "").strip() for row in source_rows),
        "labels_jsonl_text_written": False,
        "suggested_labels_not_auto_applied": True,
        "text_artifact_note": "input CSV may contain raw candidate text; output labels JSONL omits candidate text, source_path, and symbol",
        "recommendation": "run_validate_labels_then_apply_labels"
        if validation["ready_for_apply_labels"]
        else validation["recommendation"],
        "limits": (
            "This exports manually filled expected_label values from a local CSV into prompt-safe JSONL. "
            "It does not use suggested_label as a gold label, does not apply labels, does not import a gold set, "
            "does not call a local judge or provider API, and does not prove semantic quality, cache, latency, "
            "cost, or task success."
        ),
    }


def _validate_labels_summary(
    *,
    input_path: str,
    rows: Sequence[Mapping[str, Any]],
    require_complete: bool,
    min_samples: int,
    min_label_count: int,
) -> dict[str, Any]:
    row_keys = [_jsonl_label_key(row) for row in rows]
    key_counts = Counter(key for key in row_keys if key is not None)
    duplicate_key_count = sum(1 for count in key_counts.values() if count > 1)
    rows_missing_id = sum(1 for row in rows if not str(row.get("id") or "").strip())
    rows_missing_text_hash = sum(1 for row in rows if not str(row.get("text_hash") or "").strip())
    labels = [str(row.get("expected_label") or row.get("label") or "").strip().lower() for row in rows]
    expected_counts = Counter(label for label in labels if label in ALLOWED_LABELS)
    pending_label_count = sum(1 for label in labels if not label)
    invalid_label_count = sum(1 for label in labels if label and label not in ALLOWED_LABELS)
    valid_label_count = sum(expected_counts.values())
    source_label_audit = _source_expected_label_audit(rows=rows, labels=labels)
    labels_with_min_count = sum(1 for label in ALLOWED_LABELS if expected_counts.get(label, 0) >= min_label_count)
    structure_ready = bool(
        len(rows) > 0
        and rows_missing_id == 0
        and rows_missing_text_hash == 0
        and duplicate_key_count == 0
        and invalid_label_count == 0
    )
    complete_ready = bool(structure_ready and pending_label_count == 0)
    sample_ready = bool(valid_label_count >= min_samples)
    label_balance_ready = bool(labels_with_min_count == len(ALLOWED_LABELS))
    ready_for_apply = bool(structure_ready and (complete_ready or not require_complete))
    return {
        "schema_version": "prefix-local-judge-goldset-labels-validation-summary-v1",
        "prompt_safe_summary": True,
        "input_path": input_path,
        "require_complete": require_complete,
        "row_count": len(rows),
        "valid_label_row_count": valid_label_count,
        "expected_label_pending_count": pending_label_count,
        "invalid_label_row_count": invalid_label_count,
        "rows_missing_id_count": rows_missing_id,
        "rows_missing_text_hash_count": rows_missing_text_hash,
        "duplicate_label_key_count": duplicate_key_count,
        "expected_label_counts": dict(sorted(expected_counts.items())),
        "source_label_counts": _any_value_counts(rows, "source_label"),
        "source_expected_label_match_count": source_label_audit["match_count"],
        "source_expected_label_mismatch_count": source_label_audit["mismatch_count"],
        "source_expected_label_audited_count": source_label_audit["audited_count"],
        "source_expected_label_match_rate": source_label_audit["match_rate"],
        "source_expected_label_mismatch_counts": source_label_audit["mismatch_counts"],
        "possible_rule_self_confirmation": source_label_audit["possible_rule_self_confirmation"],
        "gold_labels_independent_from_rules_ready": source_label_audit["independent_from_rules_ready"],
        "semantic_hint_counts": _any_value_counts(rows, "semantic_hint"),
        "risk_tag_counts": _any_list_counts(rows, "risk_tags"),
        "label_status_counts": _any_value_counts(rows, "label_status"),
        "label_options": sorted(ALLOWED_LABELS),
        "min_samples": min_samples,
        "min_label_count": min_label_count,
        "labels_with_min_count": labels_with_min_count,
        "structure_ready": structure_ready,
        "complete_labels_ready": complete_ready,
        "sample_count_ready": sample_ready,
        "label_balance_ready": label_balance_ready,
        "ready_for_apply_labels": ready_for_apply,
        "ready_for_goldset_import_after_apply": bool(complete_ready and sample_ready and label_balance_ready),
        "text_artifact_note": "labels validation is prompt-safe and omits candidate text, source_path, and symbol",
        "recommendation": _validate_labels_recommendation(
            row_count=len(rows),
            rows_missing_id=rows_missing_id,
            rows_missing_text_hash=rows_missing_text_hash,
            duplicate_key_count=duplicate_key_count,
            pending_label_count=pending_label_count,
            invalid_label_count=invalid_label_count,
            require_complete=require_complete,
            sample_ready=sample_ready,
            label_balance_ready=label_balance_ready,
        ),
        "limits": (
            "This validates the prompt-safe labels JSONL only. It does not check that labels match every CSV row, "
            "does not read candidate text, does not apply labels, and does not prove semantic quality, cache, "
            "latency, cost, or task success. Source/expected label agreement is an audit signal only, not a "
            "correctness proof."
        ),
    }


def _source_expected_label_audit(
    *,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
) -> dict[str, Any]:
    match_count = 0
    mismatch_count = 0
    audited_count = 0
    mismatch_counts: Counter[str] = Counter()
    for row, expected_label in zip(rows, labels, strict=True):
        if expected_label not in ALLOWED_LABELS:
            continue
        source_label = str(row.get("source_label") or row.get("rule_label") or "").strip().lower()
        if source_label not in ALLOWED_LABELS:
            continue
        audited_count += 1
        if source_label == expected_label:
            match_count += 1
        else:
            mismatch_count += 1
            mismatch_counts[f"{source_label}->{expected_label}"] += 1
    match_rate = (match_count / audited_count) if audited_count else None
    possible_self_confirmation = bool(audited_count >= 20 and mismatch_count == 0)
    independent_ready = bool(audited_count > 0 and not possible_self_confirmation)
    return {
        "match_count": match_count,
        "mismatch_count": mismatch_count,
        "audited_count": audited_count,
        "match_rate": match_rate,
        "mismatch_counts": dict(sorted(mismatch_counts.items())),
        "possible_rule_self_confirmation": possible_self_confirmation,
        "independent_from_rules_ready": independent_ready,
    }


def _suggestion_row(
    row: Mapping[str, str],
    *,
    index: int,
    config: LocalModelJudgeConfig,
    min_confidence: float,
) -> dict[str, Any]:
    candidate = _candidate_row_from_csv(row, index=index)
    rule_labeled = _label_row_by_rules(candidate, min_confidence=min_confidence)
    model_labeled = _label_row_by_local_model(
        candidate,
        rule_labeled=rule_labeled,
        min_confidence=min_confidence,
        config=config,
    )
    text = str(row.get("text") or "")
    risk_tags = _parse_list(row.get("risk_tags"))
    suggested_label = _valid_label(model_labeled.get("label")) or "review"
    model_label = _valid_label(model_labeled.get("model_label"))
    rule_label = _valid_label(rule_labeled.get("label")) or _valid_label(row.get("rule_label")) or None
    source_label = _valid_label(row.get("source_label")) or None
    label_reason = str(model_labeled.get("label_reason") or "")
    return {
        "schema_version": "prefix-local-judge-goldset-label-suggestion-v1",
        "id": str(row.get("id") or f"gold-row-{index}").strip() or f"gold-row-{index}",
        "text_hash": str(row.get("text_hash") or _text_hash(text)).strip(),
        "suggested_label": suggested_label,
        "model_label": model_label,
        "model_confidence": model_labeled.get("model_confidence"),
        "confidence_meets_min": bool(
            isinstance(model_labeled.get("model_confidence"), (int, float))
            and float(model_labeled.get("model_confidence")) >= min_confidence
        ),
        "rule_label": rule_label,
        "source_label": source_label,
        "local_judge_action": model_labeled.get("local_judge_action"),
        "label_reason_code": _safe_reason_code(label_reason),
        "static_safety_clamped": label_reason.startswith("model_accept_clamped_"),
        "semantic_hint": row.get("semantic_hint") or None,
        "risk_tags": risk_tags,
        "risk_tag_count": len(risk_tags),
        "char_count": len(text),
        "line_count": text.count("\n") + 1 if text else 0,
        "existing_expected_label": _valid_label(row.get("expected_label")) or None,
        "recommended_action": "use_as_manual_review_input_not_gold_label",
    }


def _candidate_row_from_csv(row: Mapping[str, str], *, index: int) -> dict[str, Any]:
    text = str(row.get("text") or "")
    risk_tags = _parse_list(row.get("risk_tags"))
    return {
        "schema_version": "prefix-semantic-candidate-v1",
        "candidate_id": str(row.get("id") or f"gold-row-{index}"),
        "source_path": row.get("source_path") or "local_goldset_annotation_csv",
        "symbol": row.get("symbol") or None,
        "parent_hash": row.get("parent_hash") or None,
        "text_hash": row.get("text_hash") or _text_hash(text),
        "semantic_hint": row.get("semantic_hint") or "goldset_annotation",
        "confidence": _float(row.get("classifier_confidence") or row.get("confidence") or 1.0),
        "risk_tags": risk_tags,
        "char_count": len(text),
        "line_count": text.count("\n") + 1 if text else 0,
        "text": text,
    }


def _suggestions_summary(
    *,
    input_path: str,
    output_path: str,
    base_url: str,
    model: str,
    api_key_configured: bool,
    timeout: float,
    min_confidence: float,
    max_rows: int | None,
    source_row_count: int,
    suggestions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    suggested_counts = _any_value_counts(suggestions, "suggested_label")
    model_counts = _any_value_counts(suggestions, "model_label")
    rule_counts = _any_value_counts(suggestions, "rule_label")
    source_counts = _any_value_counts(suggestions, "source_label")
    action_counts = _any_value_counts(suggestions, "local_judge_action")
    reason_counts = _any_value_counts(suggestions, "label_reason_code")
    error_count = sum(1 for row in suggestions if str(row.get("label_reason_code") or "").startswith("local_judge_error"))
    missing_text_count = sum(1 for row in suggestions if _num(row.get("char_count")) <= 0)
    static_safety_clamp_count = sum(1 for row in suggestions if row.get("static_safety_clamped") is True)
    model_called_count = sum(1 for row in suggestions if row.get("local_judge_action") == "model_called")
    ready = bool(len(suggestions) > 0 and error_count == 0 and missing_text_count == 0)
    return {
        "schema_version": "prefix-local-judge-goldset-label-suggestions-summary-v1",
        "prompt_safe_summary": True,
        "input_path": input_path,
        "output_path": output_path,
        "output_written": True,
        "base_url": base_url,
        "model": model,
        "api_key_configured": api_key_configured,
        "timeout_seconds": timeout,
        "min_confidence": min_confidence,
        "max_rows": max_rows,
        "source_row_count": source_row_count,
        "suggestion_count": len(suggestions),
        "model_called_count": model_called_count,
        "local_judge_error_count": error_count,
        "rows_missing_text_count": missing_text_count,
        "static_safety_clamp_count": static_safety_clamp_count,
        "suggested_label_counts": suggested_counts,
        "model_label_counts": model_counts,
        "rule_label_counts": rule_counts,
        "source_label_counts": source_counts,
        "local_judge_action_counts": action_counts,
        "label_reason_code_counts": reason_counts,
        "model_to_suggested_label_counts": _transition_counts(suggestions, "model_label", "suggested_label"),
        "rule_to_suggested_label_counts": _transition_counts(suggestions, "rule_label", "suggested_label"),
        "source_to_suggested_label_counts": _transition_counts(suggestions, "source_label", "suggested_label"),
        "suggestions_ready_for_manual_review": ready,
        "ready_for_apply_labels": False,
        "ready_for_goldset_import_after_apply": False,
        "text_artifact_note": "input CSV contains raw candidate text; suggestions JSONL omits candidate text, source_path, and symbol",
        "recommendation": "manually_review_suggestions_then_fill_expected_label_values"
        if ready
        else "fix_local_judge_suggestions_before_manual_labeling",
        "limits": (
            "These are local-model suggestions for manual annotation only. They are not human gold labels, "
            "are not applied to expected_label, and do not prove semantic quality, cache, latency, cost, or task success."
        ),
    }


def _validate_labels_recommendation(
    *,
    row_count: int,
    rows_missing_id: int,
    rows_missing_text_hash: int,
    duplicate_key_count: int,
    pending_label_count: int,
    invalid_label_count: int,
    require_complete: bool,
    sample_ready: bool,
    label_balance_ready: bool,
) -> str:
    if row_count == 0:
        return "generate_labels_template_before_validation"
    if rows_missing_id > 0 or rows_missing_text_hash > 0:
        return "restore_label_id_and_text_hash_values"
    if duplicate_key_count > 0:
        return "deduplicate_label_id_text_hash_rows"
    if invalid_label_count > 0:
        return "fix_invalid_expected_label_values"
    if pending_label_count > 0:
        return "fill_remaining_expected_label_values" if require_complete else "fill_labels_or_apply_partial_with_caution"
    if not sample_ready:
        return "add_more_valid_label_rows_before_quality_eval"
    if not label_balance_ready:
        return "add_label_balance_before_quality_eval"
    return "run_apply_labels_require_complete"


def _merge_suggestion_rows(
    *,
    rows: Sequence[Mapping[str, str]],
    suggestions: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, str], ...], dict[str, Any]]:
    suggestion_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    duplicate_suggestion_key_count = 0
    suggestions_missing_key_count = 0
    for suggestion in suggestions:
        key = _jsonl_label_key(suggestion)
        if key is None:
            suggestions_missing_key_count += 1
            continue
        if key in suggestion_by_key:
            duplicate_suggestion_key_count += 1
            continue
        suggestion_by_key[key] = suggestion

    merged: list[dict[str, str]] = []
    matched_count = 0
    missing_suggestion_count = 0
    duplicate_csv_key_count = 0
    seen_csv_keys: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        key = _csv_label_key(row, index=index)
        if key in seen_csv_keys:
            duplicate_csv_key_count += 1
        seen_csv_keys.add(key)
        suggestion = suggestion_by_key.get(key)
        if suggestion is None:
            missing_suggestion_count += 1
        else:
            matched_count += 1
        merged.append(_suggestion_review_row(row, suggestion=suggestion))

    unmatched_suggestion_count = sum(1 for key in suggestion_by_key if key not in seen_csv_keys)
    diagnostics = {
        "input_row_count": len(rows),
        "suggestion_row_count": len(suggestions),
        "matched_suggestion_count": matched_count,
        "missing_suggestion_count": missing_suggestion_count,
        "unmatched_suggestion_count": unmatched_suggestion_count,
        "duplicate_csv_key_count": duplicate_csv_key_count,
        "duplicate_suggestion_key_count": duplicate_suggestion_key_count,
        "suggestions_missing_key_count": suggestions_missing_key_count,
    }
    return tuple(merged), diagnostics


def _suggestion_review_row(
    row: Mapping[str, str],
    *,
    suggestion: Mapping[str, Any] | None,
) -> dict[str, str]:
    suggestion = suggestion or {}
    result: dict[str, str] = {}
    for column in SUGGESTION_REVIEW_COLUMNS:
        if column == "suggested_label":
            result[column] = _valid_label(suggestion.get("suggested_label")) or ""
        elif column == "suggestion_notes":
            result[column] = _suggestion_notes(suggestion)
        elif column == "model_label":
            result[column] = _valid_label(suggestion.get("model_label")) or str(row.get("model_label") or "")
        elif column == "model_confidence":
            result[column] = "" if suggestion.get("model_confidence") is None else str(suggestion.get("model_confidence"))
        elif column == "confidence_meets_min":
            result[column] = _bool_text(suggestion.get("confidence_meets_min"))
        elif column == "static_safety_clamped":
            result[column] = _bool_text(suggestion.get("static_safety_clamped"))
        elif column == "local_judge_action":
            result[column] = str(suggestion.get("local_judge_action") or "")
        elif column == "label_reason_code":
            result[column] = str(suggestion.get("label_reason_code") or "")
        elif column == "expected_label":
            result[column] = str(row.get("expected_label") or "").strip().lower()
        elif column == "risk_tags":
            result[column] = json.dumps(_parse_list(row.get("risk_tags")), ensure_ascii=False)
        elif column in row:
            result[column] = str(row.get(column) or "")
        else:
            result[column] = ""
    return result


def _suggestion_notes(suggestion: Mapping[str, Any]) -> str:
    if not suggestion:
        return "missing_suggestion"
    return str(suggestion.get("recommended_action") or "use_as_manual_review_input_not_gold_label")


def _bool_text(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def _merge_suggestions_summary(
    *,
    input_path: str,
    suggestions_path: str,
    output_path: str,
    rows: Sequence[Mapping[str, str]],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    critical_error_count = sum(
        _num(diagnostics.get(field))
        for field in (
            "missing_suggestion_count",
            "unmatched_suggestion_count",
            "duplicate_csv_key_count",
            "duplicate_suggestion_key_count",
            "suggestions_missing_key_count",
        )
    )
    suggested_counts = Counter(_valid_label(row.get("suggested_label")) for row in rows)
    suggested_counts.pop("", None)
    expected_counts = Counter(_valid_label(row.get("expected_label")) for row in rows)
    expected_counts.pop("", None)
    ready = bool(len(rows) > 0 and critical_error_count == 0)
    return {
        "schema_version": "prefix-local-judge-goldset-suggestion-review-csv-summary-v1",
        "prompt_safe_summary": True,
        "input_path": input_path,
        "suggestions_path": suggestions_path,
        "output_path": output_path,
        "output_written": True,
        "row_count": len(rows),
        "suggestion_row_count": _num(diagnostics.get("suggestion_row_count")),
        "matched_suggestion_count": _num(diagnostics.get("matched_suggestion_count")),
        "missing_suggestion_count": _num(diagnostics.get("missing_suggestion_count")),
        "unmatched_suggestion_count": _num(diagnostics.get("unmatched_suggestion_count")),
        "duplicate_csv_key_count": _num(diagnostics.get("duplicate_csv_key_count")),
        "duplicate_suggestion_key_count": _num(diagnostics.get("duplicate_suggestion_key_count")),
        "suggestions_missing_key_count": _num(diagnostics.get("suggestions_missing_key_count")),
        "expected_label_pending_count": sum(1 for row in rows if not str(row.get("expected_label") or "").strip()),
        "expected_label_counts": dict(sorted(expected_counts.items())),
        "suggested_label_counts": dict(sorted(suggested_counts.items())),
        "suggestion_matches_source_label_counts": _transition_counts(rows, "source_label", "suggested_label"),
        "ready_for_manual_review": ready,
        "ready_for_apply_labels": False,
        "ready_for_goldset_import_after_apply": False,
        "csv_text_written": True,
        "text_artifact_note": "output CSV contains raw candidate text plus local-model suggestions; keep it local and do not commit",
        "recommendation": "manually_fill_expected_label_using_review_csv"
        if ready
        else "fix_suggestion_review_csv_join_before_manual_labeling",
        "limits": (
            "This creates a local manual-review CSV only. It does not auto-fill expected_label, "
            "does not import a gold set, does not call a provider API, and does not prove semantic quality, "
            "cache, latency, cost, or task success."
        ),
    }


def _apply_labels_summary(
    *,
    input_path: str,
    labels_path: str,
    output_path: str,
    rows: Sequence[Mapping[str, str]],
    diagnostics: Mapping[str, Any],
    require_complete: bool,
) -> dict[str, Any]:
    expected_labels = [_valid_label(row.get("expected_label")) for row in rows]
    expected_counts = Counter(label for label in expected_labels if label)
    pending_label_count = sum(1 for row in rows if not str(row.get("expected_label") or "").strip())
    invalid_label_count = sum(
        1
        for row in rows
        if str(row.get("expected_label") or "").strip() and not _valid_label(row.get("expected_label"))
    )
    rows_with_text = sum(1 for row in rows if str(row.get("text") or "").strip())
    rows_missing_text = len(rows) - rows_with_text
    critical_error_count = sum(
        _num(diagnostics.get(field))
        for field in (
            "pending_label_row_count",
            "invalid_label_row_count",
            "missing_label_key_count",
            "duplicate_label_key_count",
            "unmatched_label_count",
            "duplicate_csv_key_count",
        )
    )
    complete_after_apply = bool(len(rows) > 0 and pending_label_count == 0 and invalid_label_count == 0 and rows_missing_text == 0)
    output_written = bool(
        _num(diagnostics.get("label_row_count")) > 0
        and critical_error_count == 0
        and (complete_after_apply or not require_complete)
    )
    return {
        "schema_version": "prefix-local-judge-goldset-label-apply-summary-v1",
        "prompt_safe_summary": True,
        "input_path": input_path,
        "labels_path": labels_path,
        "output_path": output_path,
        "require_complete": require_complete,
        "output_written": output_written,
        "row_count": len(rows),
        "rows_with_text_count": rows_with_text,
        "rows_missing_text_count": rows_missing_text,
        "rows_with_expected_label_count": sum(1 for label in expected_labels if label),
        "expected_label_pending_count": pending_label_count,
        "invalid_expected_label_count": invalid_label_count,
        "expected_label_counts": dict(sorted(expected_counts.items())),
        "label_options": sorted(ALLOWED_LABELS),
        "ready_for_labeled_jsonl_import": complete_after_apply,
        "label_row_count": _num(diagnostics.get("label_row_count")),
        "valid_label_row_count": _num(diagnostics.get("valid_label_row_count")),
        "pending_label_row_count": _num(diagnostics.get("pending_label_row_count")),
        "invalid_label_row_count": _num(diagnostics.get("invalid_label_row_count")),
        "missing_label_key_count": _num(diagnostics.get("missing_label_key_count")),
        "duplicate_label_key_count": _num(diagnostics.get("duplicate_label_key_count")),
        "unmatched_label_count": _num(diagnostics.get("unmatched_label_count")),
        "duplicate_csv_key_count": _num(diagnostics.get("duplicate_csv_key_count")),
        "matched_label_count": _num(diagnostics.get("matched_label_count")),
        "label_expected_counts": dict(diagnostics.get("label_expected_counts") or {}),
        "text_artifact_note": "input/output CSV files contain raw candidate text; keep them local and do not commit",
        "recommendation": _apply_labels_recommendation(
            output_written=output_written,
            label_row_count=_num(diagnostics.get("label_row_count")),
            critical_error_count=critical_error_count,
            require_complete=require_complete,
            complete_after_apply=complete_after_apply,
            pending_label_count=pending_label_count,
            invalid_label_count=invalid_label_count,
            rows_missing_text=rows_missing_text,
        ),
        "limits": (
            "This applies local human labels by id/text_hash only. It does not call a local judge, "
            "provider API, or prove semantic quality, cache, latency, cost, or task success."
        ),
    }


def _apply_labels_recommendation(
    *,
    output_written: bool,
    label_row_count: int,
    critical_error_count: int,
    require_complete: bool,
    complete_after_apply: bool,
    pending_label_count: int,
    invalid_label_count: int,
    rows_missing_text: int,
) -> str:
    if output_written and complete_after_apply:
        return "import_labeled_csv_then_validate_goldset"
    if label_row_count == 0:
        return "fill_prompt_safe_label_jsonl_before_applying"
    if critical_error_count > 0:
        return "fix_label_jsonl_before_applying"
    if rows_missing_text > 0:
        return "restore_candidate_text_before_import"
    if invalid_label_count > 0:
        return "fix_invalid_expected_label_values"
    if pending_label_count > 0:
        return "fill_remaining_expected_label_values"
    if require_complete:
        return "rerun_apply_labels_after_labels_are_complete"
    return "inspect_apply_labels_summary"


def _annotation_worklist_rows(rows: Sequence[Mapping[str, str]]) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(rows, start=1):
        raw_label = str(row.get("expected_label") or "").strip()
        valid_label = _valid_label(raw_label)
        if valid_label:
            continue
        label_status = "invalid" if raw_label else "pending"
        item: dict[str, Any] = {
            "schema_version": "prefix-local-judge-goldset-annotation-worklist-item-v1",
            "csv_row_number": index + 1,
            "id": str(row.get("id") or f"gold-row-{index - 1}").strip() or f"gold-row-{index - 1}",
            "label_status": label_status,
            "expected_label": raw_label if label_status == "invalid" else None,
            "label_options": _parse_list(row.get("label_options")) or sorted(ALLOWED_LABELS),
            "source_label": row.get("source_label") or None,
            "rule_label": row.get("rule_label") or None,
            "model_label": row.get("model_label") or None,
            "semantic_hint": row.get("semantic_hint") or None,
            "risk_tags": _parse_list(row.get("risk_tags")),
            "review_priority": row.get("review_priority") or None,
            "recommended_action": row.get("recommended_action") or "fill_expected_label",
            "extraction": row.get("extraction") or None,
            "parent_hash": row.get("parent_hash") or None,
            "text_hash": row.get("text_hash") or _text_hash(row.get("text", "")),
        }
        yield {key: value for key, value in item.items() if value not in (None, "", [])}


def _csv_label_key(row: Mapping[str, str], *, index: int) -> tuple[str, str]:
    row_id = str(row.get("id") or f"gold-row-{index}").strip() or f"gold-row-{index}"
    text_hash = str(row.get("text_hash") or "").strip() or _text_hash(row.get("text", ""))
    return row_id, text_hash


def _jsonl_label_key(row: Mapping[str, Any]) -> tuple[str, str] | None:
    row_id = str(row.get("id") or "").strip()
    text_hash = str(row.get("text_hash") or "").strip()
    if not row_id or not text_hash:
        return None
    return row_id, text_hash


def _value_counts(rows: Sequence[Mapping[str, str]], field: str) -> dict[str, int]:
    counts = Counter(_bucket(row.get(field)) for row in rows)
    return dict(sorted(counts.items()))


def _any_value_counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(_bucket(row.get(field)) for row in rows)
    return dict(sorted(counts.items()))


def _list_counts(rows: Sequence[Mapping[str, str]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        values = _parse_list(row.get(field))
        if not values:
            counts["unknown"] += 1
            continue
        counts.update(_bucket(value) for value in values)
    return dict(sorted(counts.items()))


def _any_list_counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        values = _parse_list(row.get(field))
        if not values:
            counts["unknown"] += 1
            continue
        counts.update(_bucket(value) for value in values)
    return dict(sorted(counts.items()))


def _bucket(value: Any) -> str:
    return str(value or "").strip() or "unknown"


def _render_annotation_guide(summary: Mapping[str, Any]) -> str:
    label_options = summary.get("label_options") if isinstance(summary.get("label_options"), list) else sorted(ALLOWED_LABELS)
    lines = [
        "# Local Judge Gold-Set Annotation Guide",
        "",
        "This guide is prompt-safe. It describes how to label the local CSV but does not include candidate text.",
        "",
        "## Progress",
        "",
        f"- Input CSV: `{summary.get('input_path')}`",
        f"- Rows: `{summary.get('row_count')}`",
        f"- Filled expected_label rows: `{summary.get('rows_with_expected_label_count')}`",
        f"- Pending expected_label rows: `{summary.get('expected_label_pending_count')}`",
        f"- Invalid expected_label rows: `{summary.get('invalid_expected_label_count')}`",
        f"- Rows missing candidate text: `{summary.get('rows_missing_text_count')}`",
        f"- Current recommendation: `{summary.get('recommendation')}`",
        "",
        "## Coverage Diagnostics",
        "",
        f"- Source labels: `{_json_for_guide(summary.get('source_label_counts'))}`",
        f"- Semantic hints: `{_json_for_guide(summary.get('semantic_hint_counts'))}`",
        f"- Risk tags: `{_json_for_guide(summary.get('risk_tag_counts'))}`",
        f"- Pending source labels: `{_json_for_guide(summary.get('pending_source_label_counts'))}`",
        f"- Pending semantic hints: `{_json_for_guide(summary.get('pending_semantic_hint_counts'))}`",
        f"- Pending risk tags: `{_json_for_guide(summary.get('pending_risk_tag_counts'))}`",
        "",
        "## Label Set",
        "",
    ]
    lines.extend(f"- `{label}`" for label in label_options)
    lines.extend(
        [
            "",
            "## Labeling Rules",
            "",
            "- `accept`: the candidate is stable shared instruction/context that can be promoted without changing agent identity, tool bindings, private state, turn-specific meaning, or user constraints.",
            "- `review`: the candidate might be reusable but has uncertain boundaries, agent identity coupling, conditional wording, or low confidence. Keep these out of automatic promotion until local-judge/manual evidence supports them.",
            "- `reject`: the candidate is private, turn-specific, tool-result-bound, agent-local, or otherwise unsafe to promote into a shared reusable prefix.",
            "",
            "## Handling Rules",
            "",
            "- Fill only the `expected_label` column with one of the allowed labels.",
            "- Use `annotation_notes` for short local notes when a label is ambiguous.",
            "- Keep the CSV local because it contains raw candidate text.",
            "- Do not commit the CSV or the labeled local JSONL.",
            "- After labels are complete, run the manifest's CSV import command, then run `local_judge_goldset_validate` before any local judge quality evaluation.",
            "",
            "## Limits",
            "",
            "This guide does not prove semantic-model quality or provider cache, latency, cost, or task-success gains.",
            "",
        ]
    )
    return "\n".join(lines)


def _json_for_guide(value: Any) -> str:
    return json.dumps(value if isinstance(value, Mapping) else {}, ensure_ascii=False, sort_keys=True)


def _import_summary(
    *,
    input_path: str,
    output_path: str,
    rows: Sequence[Mapping[str, Any]],
    allow_incomplete_output: bool,
) -> dict[str, Any]:
    expected_counts = Counter(str(row.get("expected_label") or "") for row in rows)
    expected_counts.pop("", None)
    pending_label_count = sum(1 for row in rows if not row.get("expected_label"))
    invalid_label_count = sum(
        1
        for row in rows
        if row.get("expected_label") and row.get("expected_label") not in ALLOWED_LABELS
    )
    rows_with_text = sum(1 for row in rows if isinstance(row.get("text"), str) and bool(str(row.get("text")).strip()))
    rows_missing_text = len(rows) - rows_with_text
    ready = bool(len(rows) > 0 and pending_label_count == 0 and invalid_label_count == 0 and rows_missing_text == 0)
    return {
        "schema_version": "prefix-local-judge-goldset-csv-import-summary-v1",
        "prompt_safe_summary": True,
        "input_path": input_path,
        "output_path": output_path,
        "output_written": ready or allow_incomplete_output,
        "allow_incomplete_output": allow_incomplete_output,
        "gold_text_written": rows_with_text > 0,
        "row_count": len(rows),
        "rows_with_text_count": rows_with_text,
        "rows_missing_text_count": rows_missing_text,
        "expected_label_counts": dict(sorted(expected_counts.items())),
        "expected_label_pending_count": pending_label_count,
        "invalid_expected_label_count": invalid_label_count,
        "label_options": sorted(ALLOWED_LABELS),
        "ready_for_local_judge_goldset_validate": ready,
        "recommendation": _import_recommendation(
            ready=ready,
            row_count=len(rows),
            pending_label_count=pending_label_count,
            invalid_label_count=invalid_label_count,
            rows_missing_text=rows_missing_text,
        ),
        "limits": (
            "This imports local human labels only. It does not call a local judge, provider API, "
            "or prove semantic quality, cache, latency, cost, or task success."
        ),
    }


def _import_recommendation(
    *,
    ready: bool,
    row_count: int,
    pending_label_count: int,
    invalid_label_count: int,
    rows_missing_text: int,
) -> str:
    if ready:
        return "run_local_judge_goldset_validate"
    if row_count == 0:
        return "add_rows_before_importing_labeled_goldset"
    if pending_label_count > 0:
        return "fill_expected_label_in_csv_before_import"
    if invalid_label_count > 0:
        return "fix_invalid_expected_label_values"
    if rows_missing_text > 0:
        return "restore_candidate_text_before_import"
    return "inspect_imported_goldset_summary"


def _valid_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    return label if label in ALLOWED_LABELS else ""


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _optional_bool(value: Any) -> bool | None:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _normalize_optional_nonnegative_int(value: int | None) -> int | None:
    if value is None:
        return None
    return max(0, int(value))


def _transition_counts(
    rows: Sequence[Mapping[str, Any]],
    from_field: str,
    to_field: str,
) -> dict[str, int]:
    counts = Counter(
        f"{_bucket(row.get(from_field))}->{_bucket(row.get(to_field))}"
        for row in rows
        if row.get(from_field) not in (None, "") or row.get(to_field) not in (None, "")
    )
    return dict(sorted(counts.items()))


def _safe_reason_code(value: Any) -> str:
    cleaned = str(value or "none").strip().lower()
    cleaned = "".join(char if char.isalnum() or char in "._:-" else "_" for char in cleaned)
    return cleaned[:120] or "none"


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_suggestion_review_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUGGESTION_REVIEW_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_json(path: str | Path | None, value: Mapping[str, Any]) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _text_hash(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _num(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
