from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .candidate_label_eval import ALLOWED_LABELS
from .ir import stable_hash


@dataclass(frozen=True)
class LocalJudgeGoldsetTemplateResult:
    input_paths: tuple[str, ...]
    output_path: str
    summary_path: str | None
    summary: dict[str, Any]


def build_local_judge_goldset_template(
    *,
    input_paths: Sequence[str | Path],
    source_prompt_paths: Sequence[str | Path] | None = None,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    include_text: bool = False,
    require_text: bool = False,
    max_rows: int | None = None,
    seed: int = 20260605,
    strategy: str = "stratified",
    labels: Sequence[str] | None = None,
    semantic_hints: Sequence[str] | None = None,
    risk_tags: Sequence[str] | None = None,
    source_contains: Sequence[str] | None = None,
) -> LocalJudgeGoldsetTemplateResult:
    if not input_paths:
        raise ValueError("at least one input path is required")
    normalized_strategy = _normalize_strategy(strategy)
    source_paths = tuple(Path(path) for path in input_paths)
    prompt_paths = tuple(Path(path) for path in source_prompt_paths or ())
    recovery_index = _build_source_prompt_recovery_index(prompt_paths)
    rows = tuple(
        _normalize_row(
            row,
            input_path=path,
            source_row_index=index,
            recovery_index=recovery_index,
        )
        for path in source_paths
        for index, row in enumerate(_load_jsonl(path))
    )
    filtered = tuple(
        row
        for row in rows
        if _matches_filters(
            row,
            labels=labels,
            semantic_hints=semantic_hints,
            risk_tags=risk_tags,
            source_contains=source_contains,
        )
        and (not require_text or bool(row.get("text")))
    )
    selected = _select_rows(filtered, max_rows=max_rows, seed=seed, strategy=normalized_strategy)
    output_rows = tuple(
        _template_row(row, output_index=index, include_text=include_text)
        for index, row in enumerate(selected, start=1)
    )
    output_target = Path(output_path)
    _write_jsonl(output_target, output_rows)
    summary = _summary(
        input_paths=tuple(str(path) for path in source_paths),
        source_prompt_paths=tuple(str(path) for path in prompt_paths),
        recovery_index=recovery_index,
        output_path=str(output_target),
        rows=rows,
        filtered=filtered,
        selected=selected,
        output_rows=output_rows,
        include_text=include_text,
        require_text=require_text,
        max_rows=max_rows,
        seed=seed,
        strategy=normalized_strategy,
        labels=labels,
        semantic_hints=semantic_hints,
        risk_tags=risk_tags,
        source_contains=source_contains,
    )
    summary_target = _write_json(Path(summary_path), summary) if summary_path is not None else None
    return LocalJudgeGoldsetTemplateResult(
        input_paths=tuple(str(path) for path in source_paths),
        output_path=str(output_target),
        summary_path=str(summary_target) if summary_target else None,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a prompt-safe local-judge gold-set labeling template from semantic candidates "
            "or review worklists."
        )
    )
    parser.add_argument("--input", action="append", required=True, help="Input candidate/worklist JSONL. Repeatable.")
    parser.add_argument(
        "--source-prompts",
        action="append",
        help=(
            "Optional source_prompts JSONL used to recover candidate text by parent_hash/text_hash. Repeatable. "
            "Recovered text is written only with --include-text."
        ),
    )
    parser.add_argument("--output", required=True, help="Output gold-set template JSONL.")
    parser.add_argument("--summary", help="Optional prompt-safe summary JSON.")
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include raw candidate text in output rows for local human annotation. Default is metadata only.",
    )
    parser.add_argument(
        "--require-text",
        action="store_true",
        help="Keep only rows that already contain candidate text. Useful with --include-text.",
    )
    parser.add_argument("--max-rows", "--sample-size", dest="max_rows", type=int)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--strategy", choices=("stratified", "deterministic"), default="stratified")
    parser.add_argument("--label", action="append", help="Optional source label filter. Repeatable.")
    parser.add_argument("--semantic-hint", action="append", help="Optional semantic_hint filter. Repeatable.")
    parser.add_argument("--risk-tag", action="append", help="Optional risk_tags filter. Repeatable.")
    parser.add_argument("--source-contains", action="append", help="Optional source_path substring filter. Repeatable.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_local_judge_goldset_template(
        input_paths=args.input,
        source_prompt_paths=args.source_prompts,
        output_path=args.output,
        summary_path=args.summary,
        include_text=args.include_text,
        require_text=args.require_text,
        max_rows=args.max_rows,
        seed=args.seed,
        strategy=args.strategy,
        labels=args.label,
        semantic_hints=args.semantic_hint,
        risk_tags=args.risk_tag,
        source_contains=args.source_contains,
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


@dataclass(frozen=True)
class _RecoveryIndex:
    source_prompt_paths: tuple[str, ...]
    segments_by_hash: Mapping[tuple[str, str], tuple[str, str]]
    row_count: int = 0
    indexed_prompt_count: int = 0
    indexed_segment_count: int = 0
    duplicate_segment_key_count: int = 0
    content_hash_mismatch_count: int = 0
    missing_system_prompt_count: int = 0

    @property
    def enabled(self) -> bool:
        return bool(self.source_prompt_paths)


def _normalize_row(
    row: Mapping[str, Any],
    *,
    input_path: Path,
    source_row_index: int,
    recovery_index: _RecoveryIndex,
) -> dict[str, Any]:
    schema = str(row.get("schema_version") or "")
    label = _source_label(row, schema=schema)
    risk_tags = tuple(sorted(str(tag) for tag in row.get("risk_tags") or () if str(tag)))
    confidence = row.get("classifier_confidence", row.get("confidence"))
    text = row.get("text") if isinstance(row.get("text"), str) else None
    text_source = "input" if text is not None else None
    recovered_text = False
    recovery_prompt_path = None
    parent_hash = row.get("parent_hash")
    text_hash = row.get("text_hash")
    if text is None and isinstance(parent_hash, str) and isinstance(text_hash, str):
        recovered = recovery_index.segments_by_hash.get((parent_hash, text_hash))
        if recovered is not None:
            text, recovery_prompt_path = recovered
            text_source = "source_prompts"
            recovered_text = True
    candidate_id = str(row.get("candidate_id") or row.get("id") or f"{input_path.stem}:{source_row_index}")
    normalized: dict[str, Any] = {
        "input_path": str(input_path),
        "source_row_index": source_row_index,
        "source_schema_version": schema or None,
        "candidate_id": candidate_id,
        "source_path": row.get("source_path"),
        "symbol": row.get("symbol"),
        "extraction": row.get("extraction"),
        "parent_hash": parent_hash,
        "text_hash": text_hash,
        "semantic_hint": row.get("semantic_hint"),
        "classifier_confidence": _float_or_none(confidence),
        "char_count": _int(row.get("char_count")),
        "line_count": _int(row.get("line_count")),
        "risk_tags": risk_tags,
        "source_label": label,
        "label_reason": row.get("label_reason"),
        "label_source": row.get("label_source"),
        "rule_label": row.get("rule_label"),
        "rule_label_reason": row.get("rule_label_reason"),
        "model_label": row.get("model_label"),
        "model_confidence": _float_or_none(row.get("model_confidence")),
        "local_judge_action": row.get("local_judge_action"),
        "review_priority": row.get("review_priority"),
        "recommended_action": row.get("recommended_action"),
        "text": text,
        "text_source": text_source,
        "text_recovered": recovered_text,
        "text_recovery_source_prompt_path": recovery_prompt_path,
    }
    return normalized


def _build_source_prompt_recovery_index(paths: Sequence[Path]) -> _RecoveryIndex:
    segments: dict[tuple[str, str], tuple[str, str]] = {}
    row_count = 0
    indexed_prompt_count = 0
    indexed_segment_count = 0
    duplicate_segment_key_count = 0
    content_hash_mismatch_count = 0
    missing_system_prompt_count = 0
    for path in paths:
        for row in _load_jsonl(path):
            row_count += 1
            prompt = _source_prompt_text(row)
            if prompt is None:
                missing_system_prompt_count += 1
                continue
            parent_hash = stable_hash({"text": prompt})
            metadata_hash = _source_prompt_metadata_hash(row)
            if metadata_hash and metadata_hash != parent_hash:
                content_hash_mismatch_count += 1
            indexed_prompt_count += 1
            for segment in _candidate_segments(prompt):
                text_hash = stable_hash({"semantic_candidate": segment})
                key = (parent_hash, text_hash)
                if key in segments:
                    duplicate_segment_key_count += 1
                    continue
                segments[key] = (segment, str(path))
                indexed_segment_count += 1
    return _RecoveryIndex(
        source_prompt_paths=tuple(str(path) for path in paths),
        segments_by_hash=segments,
        row_count=row_count,
        indexed_prompt_count=indexed_prompt_count,
        indexed_segment_count=indexed_segment_count,
        duplicate_segment_key_count=duplicate_segment_key_count,
        content_hash_mismatch_count=content_hash_mismatch_count,
        missing_system_prompt_count=missing_system_prompt_count,
    )


def _source_prompt_text(row: Mapping[str, Any]) -> str | None:
    direct_content = row.get("content")
    if isinstance(direct_content, str) and direct_content.strip():
        return direct_content.strip()
    body = row.get("body") if isinstance(row.get("body"), Mapping) else {}
    messages = body.get("messages") if isinstance(body, Mapping) else None
    if isinstance(messages, (list, tuple)):
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            if message.get("role") != "system":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _source_prompt_metadata_hash(row: Mapping[str, Any]) -> str | None:
    body = row.get("body") if isinstance(row.get("body"), Mapping) else {}
    metadata = body.get("source_prompt_metadata") if isinstance(body, Mapping) else None
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("content_hash")
    return value if isinstance(value, str) and value else None


def _candidate_segments(prompt: str) -> Iterable[str]:
    paragraphs = [paragraph.strip() for paragraph in prompt.split("\n\n") if paragraph.strip()]
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(paragraph) <= 220 or len(lines) <= 1:
            yield paragraph
            continue
        for line in lines:
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if cleaned:
                yield cleaned


def _source_label(row: Mapping[str, Any], *, schema: str) -> str:
    raw = row.get("label")
    if raw is None and schema == "prefix-review-worklist-item-v1":
        raw = "review"
    if raw is None:
        raw = row.get("rule_label")
    value = str(raw or "unlabeled").strip().lower()
    return value or "unlabeled"


def _matches_filters(
    row: Mapping[str, Any],
    *,
    labels: Sequence[str] | None,
    semantic_hints: Sequence[str] | None,
    risk_tags: Sequence[str] | None,
    source_contains: Sequence[str] | None,
) -> bool:
    label_filter = {str(label).strip().lower() for label in labels or () if str(label).strip()}
    if label_filter and str(row.get("source_label") or "").lower() not in label_filter:
        return False
    hint_filter = {str(hint).strip() for hint in semantic_hints or () if str(hint).strip()}
    if hint_filter and str(row.get("semantic_hint") or "") not in hint_filter:
        return False
    risk_filter = {str(tag).strip() for tag in risk_tags or () if str(tag).strip()}
    if risk_filter and not (set(str(tag) for tag in row.get("risk_tags") or ()) & risk_filter):
        return False
    source_filter = [str(marker) for marker in source_contains or () if str(marker)]
    if source_filter:
        source_path = str(row.get("source_path") or "")
        if not any(marker in source_path for marker in source_filter):
            return False
    return True


def _select_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_rows: int | None,
    seed: int,
    strategy: str,
) -> tuple[Mapping[str, Any], ...]:
    sorted_rows = tuple(sorted(rows, key=_base_sort_key))
    if max_rows is None:
        return sorted_rows
    limit = max(0, max_rows)
    if limit == 0:
        return ()
    if strategy == "deterministic":
        return sorted_rows[:limit]
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in sorted_rows:
        groups[_stratum_key(row)].append(row)
    for key, values in groups.items():
        random.Random(f"{seed}:{key!r}").shuffle(values)
    selected: list[Mapping[str, Any]] = []
    ordered_keys = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for key in ordered_keys:
            values = groups[key]
            if not values:
                continue
            selected.append(values.pop(0))
            if len(selected) >= limit:
                break
    return tuple(selected)


def _template_row(row: Mapping[str, Any], *, output_index: int, include_text: bool) -> dict[str, Any]:
    item: dict[str, Any] = {
        "schema_version": "prefix-local-judge-goldset-template-item-v1",
        "id": _safe_id(f"gold-{output_index:04d}:{row.get('candidate_id')}"),
        "candidate_id": row.get("candidate_id"),
        "expected_label": None,
        "label_options": sorted(ALLOWED_LABELS),
        "source_input_path": row.get("input_path"),
        "source_row_index": row.get("source_row_index"),
        "source_schema_version": row.get("source_schema_version"),
        "source_path": row.get("source_path"),
        "symbol": row.get("symbol"),
        "extraction": row.get("extraction"),
        "parent_hash": row.get("parent_hash"),
        "text_hash": row.get("text_hash"),
        "semantic_hint": row.get("semantic_hint"),
        "classifier_confidence": row.get("classifier_confidence"),
        "char_count": row.get("char_count"),
        "line_count": row.get("line_count"),
        "risk_tags": list(row.get("risk_tags") or ()),
        "source_label": row.get("source_label"),
        "label_reason": row.get("label_reason"),
        "label_source": row.get("label_source"),
        "rule_label": row.get("rule_label"),
        "rule_label_reason": row.get("rule_label_reason"),
        "model_label": row.get("model_label"),
        "model_confidence": row.get("model_confidence"),
        "local_judge_action": row.get("local_judge_action"),
        "review_priority": row.get("review_priority"),
        "recommended_action": row.get("recommended_action"),
        "text_source": row.get("text_source"),
        "text_recovered": row.get("text_recovered"),
        "text_recovery_source_prompt_path": row.get("text_recovery_source_prompt_path"),
        "annotation_notes": None,
    }
    if include_text and isinstance(row.get("text"), str):
        item["text"] = row["text"]
    nullable_template_fields = {"expected_label", "annotation_notes"}
    return {
        key: value
        for key, value in item.items()
        if value is not None or key in nullable_template_fields
    }


def _summary(
    *,
    input_paths: tuple[str, ...],
    source_prompt_paths: tuple[str, ...],
    recovery_index: _RecoveryIndex,
    output_path: str,
    rows: Sequence[Mapping[str, Any]],
    filtered: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    output_rows: Sequence[Mapping[str, Any]],
    include_text: bool,
    require_text: bool,
    max_rows: int | None,
    seed: int,
    strategy: str,
    labels: Sequence[str] | None,
    semantic_hints: Sequence[str] | None,
    risk_tags: Sequence[str] | None,
    source_contains: Sequence[str] | None,
) -> dict[str, Any]:
    selected_with_text = sum(1 for row in selected if isinstance(row.get("text"), str) and bool(row.get("text")))
    selected_recovered_text = sum(1 for row in selected if row.get("text_recovered") is True)
    filtered_recovered_text = sum(1 for row in filtered if row.get("text_recovered") is True)
    output_text_written = include_text and any("text" in row for row in output_rows)
    return {
        "schema_version": "prefix-local-judge-goldset-template-summary-v1",
        "prompt_safe_summary": True,
        "input_paths": input_paths,
        "source_prompt_paths": source_prompt_paths,
        "output_path": output_path,
        "include_text": include_text,
        "require_text": require_text,
        "gold_text_written": output_text_written,
        "real_provider_metrics_available": False,
        "max_rows": max_rows,
        "seed": seed,
        "strategy": strategy,
        "filters": {
            "labels": sorted(str(value) for value in labels or ()),
            "semantic_hints": sorted(str(value) for value in semantic_hints or ()),
            "risk_tags": sorted(str(value) for value in risk_tags or ()),
            "source_contains": sorted(str(value) for value in source_contains or ()),
        },
        "input_row_count": len(rows),
        "filtered_row_count": len(filtered),
        "filtered_recovered_text_count": filtered_recovered_text,
        "selected_count": len(selected),
        "selected_with_text_count": selected_with_text,
        "selected_recovered_text_count": selected_recovered_text,
        "selected_missing_text_count": len(selected) - selected_with_text,
        "expected_label_pending_count": len(output_rows),
        "expected_label_counts": {},
        "label_options": sorted(ALLOWED_LABELS),
        "source_label_counts": _counts(selected, "source_label"),
        "semantic_hint_counts": _counts(selected, "semantic_hint"),
        "source_schema_counts": _counts(selected, "source_schema_version"),
        "risk_tag_counts": _risk_counts(selected),
        "text_recovery": {
            "enabled": recovery_index.enabled,
            "source_prompt_path_count": len(recovery_index.source_prompt_paths),
            "source_prompt_row_count": recovery_index.row_count,
            "indexed_prompt_count": recovery_index.indexed_prompt_count,
            "indexed_segment_count": recovery_index.indexed_segment_count,
            "duplicate_segment_key_count": recovery_index.duplicate_segment_key_count,
            "content_hash_mismatch_count": recovery_index.content_hash_mismatch_count,
            "missing_system_prompt_count": recovery_index.missing_system_prompt_count,
            "filtered_recovered_text_count": filtered_recovered_text,
            "selected_recovered_text_count": selected_recovered_text,
        },
        "text_artifact_note": (
            "output JSONL contains raw candidate text; keep it local and do not commit"
            if output_text_written
            else "output JSONL is prompt-safe metadata only"
        ),
        "ready_for_local_judge_quality_eval": False,
        "manual_labeling_required": True,
        "recommendation": _recommendation(
            selected_count=len(selected),
            include_text=include_text,
            selected_missing_text_count=len(selected) - selected_with_text,
        ),
        "limits": (
            "This is a blank human-labeling template. It does not prove local-model semantic quality, "
            "provider cache, latency, cost, task success, or broad rule generalization."
        ),
    }


def _recommendation(*, selected_count: int, include_text: bool, selected_missing_text_count: int) -> str:
    if selected_count == 0:
        return "add_real_semantic_candidate_inputs_before_gold_labeling"
    if not include_text:
        return "rerun_with_include_text_for_human_labeling_then_fill_expected_label"
    if selected_missing_text_count > 0:
        return "rerun_candidate_export_with_text_or_filter_with_require_text"
    return "fill_expected_label_then_run_local_judge_quality_eval"


def _base_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, int, str, str]:
    return (
        str(row.get("source_label") or ""),
        str(row.get("semantic_hint") or ""),
        ",".join(str(tag) for tag in row.get("risk_tags") or ()),
        -_int(row.get("char_count")),
        str(row.get("source_path") or ""),
        str(row.get("candidate_id") or ""),
    )


def _stratum_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    risk_bucket = "risk" if row.get("risk_tags") else "no_risk"
    return (
        str(row.get("source_label") or "unlabeled"),
        str(row.get("semantic_hint") or "unknown"),
        risk_bucket,
    )


def _counts(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(row.get(key) or "unknown") for row in rows)
    return dict(sorted(counts.items()))


def _risk_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(str(tag) for tag in row.get("risk_tags") or ())
    return dict(sorted(counts.items()))


def _normalize_strategy(value: str) -> str:
    if value not in {"stratified", "deterministic"}:
        raise ValueError(f"unsupported strategy: {value}")
    return value


def _safe_id(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value).strip()).strip("_")
    return cleaned[:160] or "gold-row"


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
