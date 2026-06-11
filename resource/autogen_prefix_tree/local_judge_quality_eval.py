from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .candidate_label_eval import (
    ALLOWED_LABELS,
    LocalModelJudgeConfig,
    _apply_static_safety_clamp,
    _call_openai_compatible_judge,
    _parse_model_judge_result,
)


@dataclass(frozen=True)
class LocalJudgeQualityEvalResult:
    input_path: str
    output_path: str | None
    summary_path: str | None
    summary: dict[str, Any]


def run_local_judge_quality_eval(
    *,
    input_path: str | Path,
    base_url: str,
    model: str,
    api_key: str | None = None,
    timeout: float = 30.0,
    min_confidence: float = 0.66,
    min_samples: int = 20,
    min_accuracy: float = 0.75,
    min_macro_f1: float = 0.70,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> LocalJudgeQualityEvalResult:
    source_path = Path(input_path)
    rows = tuple(_load_jsonl(source_path))
    config = LocalModelJudgeConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout,
    )
    predictions = [
        _evaluate_row(
            row,
            index=index,
            config=config,
            min_confidence=min_confidence,
        )
        for index, row in enumerate(rows)
    ]
    summary = _summary(
        predictions,
        input_path=str(source_path),
        base_url=base_url,
        model=model,
        api_key_configured=bool(api_key),
        timeout=timeout,
        min_confidence=min_confidence,
        min_samples=min_samples,
        min_accuracy=min_accuracy,
        min_macro_f1=min_macro_f1,
    )
    output_target = _write_jsonl(output_path, predictions)
    summary_target = _write_json(summary_path, summary)
    return LocalJudgeQualityEvalResult(
        input_path=str(source_path),
        output_path=str(output_target) if output_target else None,
        summary_path=str(summary_target) if summary_target else None,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a local OpenAI-compatible semantic judge against a gold JSONL label set."
    )
    parser.add_argument("--input", required=True, help="Gold JSONL with text and expected_label fields.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible local judge base URL.")
    parser.add_argument("--model", required=True, help="Local judge model name.")
    parser.add_argument("--api-key-env", help="Optional environment variable containing the local judge API key.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--min-confidence", type=float, default=0.66)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-accuracy", type=float, default=0.75)
    parser.add_argument("--min-macro-f1", type=float, default=0.70)
    parser.add_argument("--output", help="Optional prompt-safe prediction JSONL path.")
    parser.add_argument("--summary", help="Optional prompt-safe summary JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    result = run_local_judge_quality_eval(
        input_path=args.input,
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        timeout=args.timeout,
        min_confidence=args.min_confidence,
        min_samples=args.min_samples,
        min_accuracy=args.min_accuracy,
        min_macro_f1=args.min_macro_f1,
        output_path=args.output,
        summary_path=args.summary,
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


def _evaluate_row(
    row: Mapping[str, Any],
    *,
    index: int,
    config: LocalModelJudgeConfig,
    min_confidence: float,
) -> dict[str, Any]:
    text = row.get("text")
    expected = str(row.get("expected_label") or row.get("gold_label") or "").strip().lower()
    base = _prediction_base(row, index=index, text=text, expected_label=expected)
    if expected not in ALLOWED_LABELS:
        return {**base, "error_type": "ValueError", "error_reason": "invalid_expected_label"}
    if not isinstance(text, str) or not text.strip():
        return {**base, "error_type": "ValueError", "error_reason": "missing_text"}
    candidate = _candidate_row(row, index=index, text=text)
    try:
        raw = _call_openai_compatible_judge(config, candidate)
        raw_label, raw_reason, confidence = _parse_model_judge_result(raw)
    except Exception as exc:  # noqa: BLE001 - each row should preserve partial evidence.
        return {**base, "error_type": type(exc).__name__, "error_reason": _safe_error_reason(exc)}
    label, label_reason = _apply_static_safety_clamp(
        candidate,
        model_label=raw_label,
        model_reason=raw_reason,
        model_confidence=confidence,
        min_confidence=min_confidence,
    )
    confidence_meets_min = confidence >= min_confidence
    return {
        **base,
        "raw_model_label": raw_label,
        "predicted_label": label,
        "model_confidence": confidence,
        "confidence_meets_min": confidence_meets_min,
        "correct": label == expected,
        "raw_model_correct": raw_label == expected,
        "model_reason_available": True,
        "label_reason": label_reason,
        "static_safety_clamped": raw_label != label,
    }


def _prediction_base(
    row: Mapping[str, Any],
    *,
    index: int,
    text: Any,
    expected_label: str,
) -> dict[str, Any]:
    text_value = text if isinstance(text, str) else ""
    risk_tags = row.get("risk_tags") if isinstance(row.get("risk_tags"), list) else []
    return {
        "schema_version": "prefix-local-judge-quality-prediction-v1",
        "input_index": index,
        "example_id": _safe_id(row.get("id") or row.get("candidate_id") or f"row-{index}"),
        "text_hash": _text_hash(text_value),
        "expected_label": expected_label or None,
        "raw_model_label": None,
        "predicted_label": None,
        "model_confidence": None,
        "confidence_meets_min": False,
        "correct": False,
        "raw_model_correct": False,
        "semantic_hint": row.get("semantic_hint"),
        "risk_tag_count": len(risk_tags),
        "char_count": len(text_value),
        "line_count": text_value.count("\n") + 1 if text_value else 0,
        "error_type": None,
        "error_reason": None,
    }


def _candidate_row(row: Mapping[str, Any], *, index: int, text: str) -> dict[str, Any]:
    risk_tags = row.get("risk_tags") if isinstance(row.get("risk_tags"), list) else []
    return {
        "schema_version": "prefix-semantic-candidate-v1",
        "candidate_id": str(row.get("id") or row.get("candidate_id") or f"gold-{index}"),
        "source_path": "local_judge_quality_eval.gold",
        "semantic_hint": row.get("semantic_hint") or "quality_eval_gold",
        "confidence": row.get("classifier_confidence", row.get("confidence", 1.0)),
        "risk_tags": risk_tags,
        "char_count": len(text),
        "line_count": text.count("\n") + 1,
        "text": text,
    }


def _summary(
    predictions: Sequence[Mapping[str, Any]],
    *,
    input_path: str,
    base_url: str,
    model: str,
    api_key_configured: bool,
    timeout: float,
    min_confidence: float,
    min_samples: int,
    min_accuracy: float,
    min_macro_f1: float,
) -> dict[str, Any]:
    valid = [
        row
        for row in predictions
        if row.get("expected_label") in ALLOWED_LABELS and row.get("error_type") is None
    ]
    attempted = [row for row in predictions if row.get("expected_label") in ALLOWED_LABELS]
    evaluated = [row for row in valid if row.get("predicted_label") in ALLOWED_LABELS]
    raw_evaluated = [row for row in valid if row.get("raw_model_label") in ALLOWED_LABELS]
    correct_count = sum(1 for row in evaluated if row.get("correct") is True)
    accuracy = _ratio_or_none(correct_count, len(evaluated))
    confusion = _confusion_matrix(evaluated)
    metrics = _per_label_metrics(confusion)
    macro_f1 = _macro_value(metrics, "f1")
    raw_correct_count = sum(1 for row in raw_evaluated if row.get("raw_model_correct") is True)
    raw_accuracy = _ratio_or_none(raw_correct_count, len(raw_evaluated))
    raw_confusion = _confusion_matrix(raw_evaluated, predicted_field="raw_model_label")
    raw_metrics = _per_label_metrics(raw_confusion)
    raw_macro_f1 = _macro_value(raw_metrics, "f1")
    parse_error_count = sum(1 for row in attempted if row.get("error_type"))
    low_confidence_count = sum(1 for row in evaluated if row.get("confidence_meets_min") is not True)
    ready = bool(
        len(evaluated) >= min_samples
        and parse_error_count == 0
        and low_confidence_count == 0
        and accuracy is not None
        and accuracy >= min_accuracy
        and macro_f1 is not None
        and macro_f1 >= min_macro_f1
    )
    expected_counts = Counter(str(row.get("expected_label")) for row in attempted if row.get("expected_label"))
    predicted_counts = Counter(str(row.get("predicted_label")) for row in evaluated if row.get("predicted_label"))
    raw_predicted_counts = Counter(str(row.get("raw_model_label")) for row in raw_evaluated if row.get("raw_model_label"))
    error_counts = Counter(str(row.get("error_reason") or "none") for row in predictions if row.get("error_type"))
    static_safety_clamp_count = sum(1 for row in evaluated if row.get("static_safety_clamped") is True)
    static_safety_clamp_corrected_count = sum(
        1
        for row in evaluated
        if row.get("static_safety_clamped") is True
        and row.get("raw_model_correct") is not True
        and row.get("correct") is True
    )
    static_safety_clamp_worsened_count = sum(
        1
        for row in evaluated
        if row.get("static_safety_clamped") is True
        and row.get("raw_model_correct") is True
        and row.get("correct") is not True
    )
    quality_diagnostics = _quality_diagnostics(
        evaluated=evaluated,
        raw_evaluated=raw_evaluated,
        metrics=metrics,
        raw_metrics=raw_metrics,
        expected_counts=expected_counts,
        predicted_counts=predicted_counts,
        raw_predicted_counts=raw_predicted_counts,
        accuracy=accuracy,
        raw_accuracy=raw_accuracy,
        macro_f1=macro_f1,
        raw_macro_f1=raw_macro_f1,
        min_accuracy=min_accuracy,
        min_macro_f1=min_macro_f1,
        parse_error_count=parse_error_count,
        low_confidence_count=low_confidence_count,
    )
    return {
        "schema_version": "prefix-local-judge-quality-eval-summary-v1",
        "prompt_safe_summary": True,
        "input_path": input_path,
        "base_url": base_url,
        "model": model,
        "api_key_configured": api_key_configured,
        "timeout_seconds": timeout,
        "min_confidence": min_confidence,
        "min_samples": min_samples,
        "min_accuracy": min_accuracy,
        "min_macro_f1": min_macro_f1,
        "sample_count": len(predictions),
        "attempted_count": len(attempted),
        "evaluated_count": len(evaluated),
        "model_called_count": len(evaluated),
        "invalid_gold_count": sum(1 for row in predictions if row.get("expected_label") not in ALLOWED_LABELS),
        "parse_error_count": parse_error_count,
        "low_confidence_count": low_confidence_count,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "raw_model_correct_count": raw_correct_count,
        "raw_model_accuracy": raw_accuracy,
        "raw_model_macro_f1": raw_macro_f1,
        "static_safety_clamp_count": static_safety_clamp_count,
        "static_safety_clamp_corrected_count": static_safety_clamp_corrected_count,
        "static_safety_clamp_worsened_count": static_safety_clamp_worsened_count,
        "static_safety_clamp_net_correct_delta": static_safety_clamp_corrected_count
        - static_safety_clamp_worsened_count,
        "expected_label_counts": dict(sorted(expected_counts.items())),
        "predicted_label_counts": dict(sorted(predicted_counts.items())),
        "raw_model_label_counts": dict(sorted(raw_predicted_counts.items())),
        "raw_model_to_predicted_label_counts": _transition_counts(
            evaluated,
            from_field="raw_model_label",
            to_field="predicted_label",
        ),
        "error_reason_counts": dict(sorted(error_counts.items())),
        "confusion_matrix": confusion,
        "per_label_metrics": metrics,
        "raw_model_confusion_matrix": raw_confusion,
        "raw_model_per_label_metrics": raw_metrics,
        "quality_diagnostics": quality_diagnostics,
        "mismatch_diagnostics": _mismatch_diagnostics(evaluated),
        "failure_mode_codes": quality_diagnostics["failure_mode_codes"],
        "semantic_quality_claim_allowed": ready,
        "static_safety_clamp_policy": {
            "schema_version": "prefix-local-judge-static-safety-clamp-v1",
            "applied_to_quality_eval": True,
            "raw_model_metrics_preserved": True,
            "description": (
                "The evaluated predicted_label matches the production local-judge suggestion path: "
                "raw model accept labels are downgraded when confidence, semantic hint, or high-risk tags "
                "violate the static safety policy."
            ),
        },
        "ready": ready,
        "semantic_quality_metrics_available": bool(evaluated),
        "real_provider_metrics_available": False,
        "gold_text_written": False,
        "recommendation": _recommendation(
            ready=ready,
            evaluated_count=len(evaluated),
            min_samples=min_samples,
            parse_error_count=parse_error_count,
            low_confidence_count=low_confidence_count,
            accuracy=accuracy,
            min_accuracy=min_accuracy,
            macro_f1=macro_f1,
            min_macro_f1=min_macro_f1,
        ),
        "limits": (
            "This evaluates local semantic judge labels against the supplied gold JSONL only. "
            "predicted_label includes the same static safety clamp used by local suggestion generation; "
            "raw_model_* fields preserve the unclamped model behavior. It does not prove provider cache, "
            "latency, cost, task success, or broad cross-framework generalization."
        ),
    }


def _quality_diagnostics(
    *,
    evaluated: Sequence[Mapping[str, Any]],
    raw_evaluated: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Mapping[str, Any]],
    raw_metrics: Mapping[str, Mapping[str, Any]],
    expected_counts: Mapping[str, int],
    predicted_counts: Mapping[str, int],
    raw_predicted_counts: Mapping[str, int],
    accuracy: float | None,
    raw_accuracy: float | None,
    macro_f1: float | None,
    raw_macro_f1: float | None,
    min_accuracy: float,
    min_macro_f1: float,
    parse_error_count: int,
    low_confidence_count: int,
) -> dict[str, Any]:
    expected_labels = sorted(label for label in ALLOWED_LABELS if int(expected_counts.get(label, 0)) > 0)
    predicted_label_total = sum(int(predicted_counts.get(label, 0)) for label in ALLOWED_LABELS)
    raw_predicted_label_total = sum(int(raw_predicted_counts.get(label, 0)) for label in ALLOWED_LABELS)
    missing_predicted_labels = [
        label for label in expected_labels if int(predicted_counts.get(label, 0)) == 0
    ]
    missing_raw_model_labels = [
        label for label in expected_labels if int(raw_predicted_counts.get(label, 0)) == 0
    ]
    per_label_recall = {
        label: _metric_number(metrics, label, "recall")
        for label in expected_labels
    }
    raw_per_label_recall = {
        label: _metric_number(raw_metrics, label, "recall")
        for label in expected_labels
    }
    zero_recall_labels = [
        label for label, recall in per_label_recall.items() if recall == 0.0
    ]
    raw_zero_recall_labels = [
        label for label, recall in raw_per_label_recall.items() if recall == 0.0
    ]
    failure_modes = _quality_failure_modes(
        accuracy=accuracy,
        raw_accuracy=raw_accuracy,
        macro_f1=macro_f1,
        raw_macro_f1=raw_macro_f1,
        min_accuracy=min_accuracy,
        min_macro_f1=min_macro_f1,
        missing_predicted_labels=missing_predicted_labels,
        missing_raw_model_labels=missing_raw_model_labels,
        zero_recall_labels=zero_recall_labels,
        raw_zero_recall_labels=raw_zero_recall_labels,
        predicted_label_total=predicted_label_total,
        raw_predicted_label_total=raw_predicted_label_total,
        predicted_counts=predicted_counts,
        raw_predicted_counts=raw_predicted_counts,
        expected_label_count=len(expected_labels),
        parse_error_count=parse_error_count,
        low_confidence_count=low_confidence_count,
    )
    return {
        "schema_version": "prefix-local-judge-quality-diagnostics-v1",
        "prompt_safe_summary": True,
        "expected_label_coverage_count": len(expected_labels),
        "predicted_label_coverage_count": _positive_label_count(predicted_counts),
        "raw_model_label_coverage_count": _positive_label_count(raw_predicted_counts),
        "missing_predicted_labels": missing_predicted_labels,
        "missing_raw_model_labels": missing_raw_model_labels,
        "per_label_recall": per_label_recall,
        "raw_model_per_label_recall": raw_per_label_recall,
        "zero_recall_labels": zero_recall_labels,
        "raw_model_zero_recall_labels": raw_zero_recall_labels,
        "predicted_label_collapse": _label_collapse(
            predicted_counts,
            total=predicted_label_total,
            expected_label_count=len(expected_labels),
        ),
        "raw_model_label_collapse": _label_collapse(
            raw_predicted_counts,
            total=raw_predicted_label_total,
            expected_label_count=len(expected_labels),
        ),
        "max_predicted_label_rate": _max_label_rate(predicted_counts, predicted_label_total),
        "raw_model_max_label_rate": _max_label_rate(raw_predicted_counts, raw_predicted_label_total),
        "accept_false_positive_count": _false_positive_count(evaluated, predicted_field="predicted_label", label="accept"),
        "raw_model_accept_false_positive_count": _false_positive_count(
            raw_evaluated,
            predicted_field="raw_model_label",
            label="accept",
        ),
        "reject_recall": per_label_recall.get("reject"),
        "raw_model_reject_recall": raw_per_label_recall.get("reject"),
        "review_recall": per_label_recall.get("review"),
        "raw_model_review_recall": raw_per_label_recall.get("review"),
        "failure_mode_codes": failure_modes,
    }


def _quality_failure_modes(
    *,
    accuracy: float | None,
    raw_accuracy: float | None,
    macro_f1: float | None,
    raw_macro_f1: float | None,
    min_accuracy: float,
    min_macro_f1: float,
    missing_predicted_labels: Sequence[str],
    missing_raw_model_labels: Sequence[str],
    zero_recall_labels: Sequence[str],
    raw_zero_recall_labels: Sequence[str],
    predicted_label_total: int,
    raw_predicted_label_total: int,
    predicted_counts: Mapping[str, int],
    raw_predicted_counts: Mapping[str, int],
    expected_label_count: int,
    parse_error_count: int,
    low_confidence_count: int,
) -> list[str]:
    modes: list[str] = []
    if parse_error_count > 0:
        modes.append("parse_errors")
    if low_confidence_count > 0:
        modes.append("low_confidence_outputs")
    if accuracy is None or accuracy < min_accuracy:
        modes.append("accuracy_below_min")
    if macro_f1 is None or macro_f1 < min_macro_f1:
        modes.append("macro_f1_below_min")
    if raw_accuracy is not None and raw_accuracy < min_accuracy:
        modes.append("raw_model_accuracy_below_min")
    if raw_macro_f1 is not None and raw_macro_f1 < min_macro_f1:
        modes.append("raw_model_macro_f1_below_min")
    if _label_collapse(predicted_counts, total=predicted_label_total, expected_label_count=expected_label_count):
        modes.append("predicted_label_collapse")
    if _label_collapse(
        raw_predicted_counts,
        total=raw_predicted_label_total,
        expected_label_count=expected_label_count,
    ):
        modes.append("raw_model_label_collapse")
    modes.extend(f"missing_{label}_predictions" for label in missing_predicted_labels)
    modes.extend(f"raw_model_missing_{label}_predictions" for label in missing_raw_model_labels)
    modes.extend(f"{label}_zero_recall" for label in zero_recall_labels)
    modes.extend(f"raw_model_{label}_zero_recall" for label in raw_zero_recall_labels)
    return _dedupe_preserve_order(modes)


def _mismatch_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mismatches = [
        row
        for row in rows
        if row.get("expected_label") in ALLOWED_LABELS
        and row.get("predicted_label") in ALLOWED_LABELS
        and row.get("expected_label") != row.get("predicted_label")
    ]
    raw_mismatches = [
        row
        for row in rows
        if row.get("expected_label") in ALLOWED_LABELS
        and row.get("raw_model_label") in ALLOWED_LABELS
        and row.get("expected_label") != row.get("raw_model_label")
    ]
    return {
        "schema_version": "prefix-local-judge-quality-mismatch-diagnostics-v1",
        "prompt_safe_summary": True,
        "mismatch_count": len(mismatches),
        "raw_model_mismatch_count": len(raw_mismatches),
        "mismatch_label_transition_counts": _transition_counts(
            mismatches,
            from_field="expected_label",
            to_field="predicted_label",
        ),
        "raw_model_mismatch_label_transition_counts": _transition_counts(
            raw_mismatches,
            from_field="expected_label",
            to_field="raw_model_label",
        ),
        "mismatch_semantic_hint_counts": _value_counts(mismatches, "semantic_hint"),
        "raw_model_mismatch_semantic_hint_counts": _value_counts(raw_mismatches, "semantic_hint"),
        "mismatch_risk_tag_count_counts": _value_counts(mismatches, "risk_tag_count"),
        "raw_model_mismatch_risk_tag_count_counts": _value_counts(raw_mismatches, "risk_tag_count"),
    }


def _metric_number(
    metrics: Mapping[str, Mapping[str, Any]],
    label: str,
    field: str,
) -> float | None:
    value = metrics.get(label, {}).get(field)
    if isinstance(value, int | float):
        return float(value)
    return None


def _positive_label_count(counts: Mapping[str, int]) -> int:
    return sum(1 for label in ALLOWED_LABELS if int(counts.get(label, 0)) > 0)


def _label_collapse(
    counts: Mapping[str, int],
    *,
    total: int,
    expected_label_count: int,
) -> bool:
    if total <= 0 or expected_label_count <= 1:
        return False
    coverage_count = _positive_label_count(counts)
    return coverage_count < expected_label_count or (_max_label_rate(counts, total) or 0.0) >= 0.9


def _max_label_rate(counts: Mapping[str, int], total: int) -> float | None:
    if total <= 0:
        return None
    return max((int(counts.get(label, 0)) for label in ALLOWED_LABELS), default=0) / total


def _false_positive_count(
    rows: Sequence[Mapping[str, Any]],
    *,
    predicted_field: str,
    label: str,
) -> int:
    return sum(
        1
        for row in rows
        if row.get(predicted_field) == label and row.get("expected_label") != label
    )


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _confusion_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    predicted_field: str = "predicted_label",
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        expected: {predicted: 0 for predicted in sorted(ALLOWED_LABELS)}
        for expected in sorted(ALLOWED_LABELS)
    }
    for row in rows:
        expected = str(row.get("expected_label"))
        predicted = str(row.get(predicted_field))
        if expected in ALLOWED_LABELS and predicted in ALLOWED_LABELS:
            matrix[expected][predicted] += 1
    return matrix


def _transition_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    from_field: str,
    to_field: str,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        source = row.get(from_field)
        target = row.get(to_field)
        if source in ALLOWED_LABELS and target in ALLOWED_LABELS:
            counts[f"{source}->{target}"] += 1
    return dict(sorted(counts.items()))


def _value_counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = row.get(field)
        if value is None:
            counts["unknown"] += 1
        else:
            counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _per_label_metrics(confusion: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    labels = sorted(ALLOWED_LABELS)
    for label in labels:
        tp = int(confusion.get(label, {}).get(label, 0))
        fp = sum(int(confusion.get(other, {}).get(label, 0)) for other in labels if other != label)
        fn = sum(int(confusion.get(label, {}).get(other, 0)) for other in labels if other != label)
        precision = _ratio_or_none(tp, tp + fp)
        recall = _ratio_or_none(tp, tp + fn)
        f1 = None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        result[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(confusion.get(label, {}).values())}
    return result


def _macro_value(metrics: Mapping[str, Mapping[str, Any]], field: str) -> float | None:
    values = [value.get(field) for value in metrics.values() if isinstance(value.get(field), (int, float))]
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _recommendation(
    *,
    ready: bool,
    evaluated_count: int,
    min_samples: int,
    parse_error_count: int,
    low_confidence_count: int,
    accuracy: float | None,
    min_accuracy: float,
    macro_f1: float | None,
    min_macro_f1: float,
) -> str:
    if ready:
        return "semantic_quality_claim_allowed_for_gold_set_only"
    if evaluated_count < min_samples:
        return "add_more_gold_examples_before_semantic_quality_claim"
    if parse_error_count > 0:
        return "fix_local_judge_json_schema_before_semantic_quality_claim"
    if low_confidence_count > 0:
        return "inspect_low_confidence_judge_outputs_before_semantic_quality_claim"
    if accuracy is None or accuracy < min_accuracy:
        return "improve_or_recalibrate_local_judge_accuracy"
    if macro_f1 is None or macro_f1 < min_macro_f1:
        return "improve_label_balance_or_macro_f1_before_semantic_quality_claim"
    return "inspect_local_judge_quality_eval"


def _write_jsonl(path: str | Path | None, rows: Sequence[Mapping[str, Any]]) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return target


def _write_json(path: str | Path | None, value: Mapping[str, Any]) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _text_hash(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_id(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value).strip()).strip("_")
    return cleaned[:120] or "row"


def _safe_error_reason(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return "response_content_not_json"
    if isinstance(exc, urllib.error.HTTPError):
        return f"http_status_{exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return "url_error"
    if isinstance(exc, KeyError):
        return "missing_response_field"
    if isinstance(exc, IndexError):
        return "missing_response_choice"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ValueError):
        return _safe_token(str(exc))
    return "local_judge_error"


def _safe_token(value: Any) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(value).strip().lower()).strip("_")
    return cleaned[:80] or "local_judge_error"


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


if __name__ == "__main__":
    raise SystemExit(main())
