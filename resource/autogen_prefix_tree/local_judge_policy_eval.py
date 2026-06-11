from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .candidate_label_eval import ALLOWED_LABELS


POLICIES = (
    "source_label",
    "rule_label",
    "raw_model_label",
    "clamped_model_label",
    "rule_review_clamped_model",
    "fail_closed_rule_and_model_accept",
)


@dataclass(frozen=True)
class LocalJudgePolicyEvalResult:
    input_path: str
    predictions_path: str | None
    summary_path: str | None
    report_path: str | None
    summary: dict[str, Any]


def run_local_judge_policy_eval(
    *,
    input_path: str | Path,
    predictions_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    report_path: str | Path | None = None,
    min_accuracy: float = 0.75,
    min_macro_f1: float = 0.70,
) -> LocalJudgePolicyEvalResult:
    source_path = Path(input_path)
    prediction_file = Path(predictions_path) if predictions_path is not None else None
    gold_rows = tuple(_load_jsonl(source_path))
    predictions = tuple(_load_jsonl(prediction_file)) if prediction_file is not None else ()
    joined = tuple(_join_rows(gold_rows, predictions))
    policy_summaries = [
        _policy_summary(
            policy_name=policy_name,
            rows=joined,
            min_accuracy=min_accuracy,
            min_macro_f1=min_macro_f1,
        )
        for policy_name in POLICIES
    ]
    best_ready = _best_policy(policy_summaries, ready_only=True)
    best_overall = _best_policy(policy_summaries, ready_only=False)
    summary = {
        "schema_version": "prefix-local-judge-policy-eval-summary-v1",
        "prompt_safe_summary": True,
        "input_path": str(source_path),
        "predictions_path": str(prediction_file) if prediction_file is not None else None,
        "row_count": len(gold_rows),
        "joined_prediction_count": sum(1 for row in joined if row.get("prediction_joined") is True),
        "missing_prediction_count": sum(1 for row in joined if row.get("prediction_joined") is not True),
        "min_accuracy": min_accuracy,
        "min_macro_f1": min_macro_f1,
        "policy_names": list(POLICIES),
        "policy_summaries": policy_summaries,
        "policy_failure_diagnostics": _policy_failure_diagnostics(policy_summaries),
        "best_ready_policy": best_ready,
        "best_overall_policy": best_overall,
        "ready": best_ready is not None,
        "semantic_quality_claim_allowed": best_ready is not None,
        "real_provider_metrics_available": False,
        "gold_text_written": False,
        "recommendation": _recommendation(best_ready=best_ready, best_overall=best_overall),
        "limits": (
            "This compares local rule/model policy labels against the supplied gold JSONL using prompt-safe "
            "metadata and prior prediction labels only. It does not call a model, read candidate text into "
            "the report, or prove provider cache, latency, cost, task success, or broad generalization."
        ),
    }
    summary_target = _write_json(summary_path, summary)
    report_target = _write_text(report_path, render_markdown_policy_eval(summary))
    return LocalJudgePolicyEvalResult(
        input_path=str(source_path),
        predictions_path=str(prediction_file) if prediction_file is not None else None,
        summary_path=str(summary_target) if summary_target else None,
        report_path=str(report_target) if report_target else None,
        summary=summary,
    )


def render_markdown_policy_eval(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Local Judge Policy Evaluation",
        "",
        f"Ready: `{_format_value(summary.get('ready'))}`",
        f"Best ready policy: `{_format_value(summary.get('best_ready_policy'))}`",
        f"Best overall policy: `{_format_value(summary.get('best_overall_policy'))}`",
        "",
        "## Policies",
        "",
        "| Policy | Ready | Accuracy | Macro F1 | Predicted Counts | Failure Modes |",
        "|---|---:|---:|---:|---|---|",
    ]
    for policy in summary.get("policy_summaries") if isinstance(summary.get("policy_summaries"), list) else []:
        if not isinstance(policy, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_value(policy.get("policy_name")),
                    _format_value(policy.get("ready")),
                    _format_value(policy.get("accuracy")),
                    _format_value(policy.get("macro_f1")),
                    _format_value(policy.get("predicted_label_counts")),
                    _format_value(policy.get("failure_mode_codes")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Failure Diagnostics",
            "",
            "| Policy | Top Transitions | Top Semantic Hints | Top Risk Tags |",
            "|---|---|---|---|",
        ]
    )
    diagnostics = summary.get("policy_failure_diagnostics")
    if isinstance(diagnostics, Mapping):
        for policy_name in POLICIES:
            diagnostic = diagnostics.get(policy_name)
            if not isinstance(diagnostic, Mapping):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        policy_name,
                        _format_value(diagnostic.get("top_mismatch_label_transitions")),
                        _format_value(diagnostic.get("top_mismatch_semantic_hints")),
                        _format_value(diagnostic.get("top_mismatch_risk_tags")),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- This report is prompt-safe and does not include candidate text or API key values.",
            "- Passing a policy eval is scoped to this gold set only; real provider metrics still require A/B runs.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare rule/model/hybrid local judge policies against a gold JSONL label set."
    )
    parser.add_argument("--input", required=True, help="Gold JSONL with expected_label and prompt-safe metadata.")
    parser.add_argument("--predictions", help="Optional local_judge_quality_eval prediction JSONL.")
    parser.add_argument("--min-accuracy", type=float, default=0.75)
    parser.add_argument("--min-macro-f1", type=float, default=0.70)
    parser.add_argument("--summary", help="Optional prompt-safe summary JSON path.")
    parser.add_argument("--report-md", help="Optional prompt-safe Markdown report path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_local_judge_policy_eval(
        input_path=args.input,
        predictions_path=args.predictions,
        summary_path=args.summary,
        report_path=args.report_md,
        min_accuracy=args.min_accuracy,
        min_macro_f1=args.min_macro_f1,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0 if result.summary.get("ready") is True else 1


def _load_jsonl(path: Path | None) -> Iterable[dict[str, Any]]:
    if path is None:
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if isinstance(value, dict):
                yield value


def _join_rows(
    gold_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> Iterable[dict[str, Any]]:
    predictions_by_index = {
        int(row["input_index"]): row
        for row in predictions
        if isinstance(row.get("input_index"), int)
    }
    for index, row in enumerate(gold_rows):
        prediction = predictions_by_index.get(index)
        yield {
            "input_index": index,
            "expected_label": _safe_label(row.get("expected_label") or row.get("gold_label")),
            "source_label": _safe_label(row.get("source_label")),
            "rule_label": _safe_label(row.get("rule_label")),
            "semantic_hint": row.get("semantic_hint"),
            "risk_tags": row.get("risk_tags") if isinstance(row.get("risk_tags"), list) else [],
            "prediction_joined": prediction is not None,
            "raw_model_label": _safe_label(prediction.get("raw_model_label")) if prediction else None,
            "clamped_model_label": _safe_label(prediction.get("predicted_label")) if prediction else None,
        }


def _policy_summary(
    *,
    policy_name: str,
    rows: Sequence[Mapping[str, Any]],
    min_accuracy: float,
    min_macro_f1: float,
) -> dict[str, Any]:
    predictions = [
        {
            **row,
            "policy_label": _policy_label(policy_name=policy_name, row=row),
        }
        for row in rows
    ]
    evaluated = [
        row
        for row in predictions
        if row.get("expected_label") in ALLOWED_LABELS and row.get("policy_label") in ALLOWED_LABELS
    ]
    confusion = _confusion_matrix(evaluated)
    metrics = _per_label_metrics(confusion)
    accuracy = _ratio_or_none(
        sum(1 for row in evaluated if row.get("expected_label") == row.get("policy_label")),
        len(evaluated),
    )
    macro_f1 = _macro_value(metrics, "f1")
    predicted_counts = Counter(str(row.get("policy_label")) for row in evaluated)
    expected_counts = Counter(str(row.get("expected_label")) for row in evaluated)
    failure_modes = _failure_modes(
        accuracy=accuracy,
        macro_f1=macro_f1,
        min_accuracy=min_accuracy,
        min_macro_f1=min_macro_f1,
        expected_counts=expected_counts,
        predicted_counts=predicted_counts,
        metrics=metrics,
    )
    ready = bool(
        evaluated
        and accuracy is not None
        and accuracy >= min_accuracy
        and macro_f1 is not None
        and macro_f1 >= min_macro_f1
        and not failure_modes
    )
    return {
        "schema_version": "prefix-local-judge-policy-summary-v1",
        "policy_name": policy_name,
        "ready": ready,
        "evaluated_count": len(evaluated),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "expected_label_counts": dict(sorted(expected_counts.items())),
        "predicted_label_counts": dict(sorted(predicted_counts.items())),
        "confusion_matrix": confusion,
        "per_label_metrics": metrics,
        "failure_mode_codes": failure_modes,
        "mismatch_diagnostics": _mismatch_diagnostics(evaluated),
        "automation_policy": _automation_policy(policy_name),
    }


def _policy_label(*, policy_name: str, row: Mapping[str, Any]) -> str | None:
    source = row.get("source_label")
    rule = row.get("rule_label")
    raw = row.get("raw_model_label")
    clamped = row.get("clamped_model_label")
    if policy_name == "source_label":
        return source if source in ALLOWED_LABELS else None
    if policy_name == "rule_label":
        return rule if rule in ALLOWED_LABELS else None
    if policy_name == "raw_model_label":
        return raw if raw in ALLOWED_LABELS else None
    if policy_name == "clamped_model_label":
        return clamped if clamped in ALLOWED_LABELS else None
    if policy_name == "rule_review_clamped_model":
        if rule == "review" and clamped in ALLOWED_LABELS:
            return clamped
        return rule if rule in ALLOWED_LABELS else None
    if policy_name == "fail_closed_rule_and_model_accept":
        if rule == "reject":
            return "reject"
        if rule == "accept" and clamped == "accept":
            return "accept"
        return "review"
    raise ValueError(f"unknown policy: {policy_name}")


def _automation_policy(policy_name: str) -> str:
    if policy_name in {"raw_model_label", "clamped_model_label", "rule_review_clamped_model"}:
        return "diagnostic_only_until_goldset_quality_gate_passes"
    if policy_name == "fail_closed_rule_and_model_accept":
        return "conservative_candidate_for_validator_gate"
    return "static_baseline"


def _confusion_matrix(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    labels = sorted(ALLOWED_LABELS)
    matrix = {expected: {predicted: 0 for predicted in labels} for expected in labels}
    for row in rows:
        expected = str(row.get("expected_label"))
        predicted = str(row.get("policy_label"))
        if expected in ALLOWED_LABELS and predicted in ALLOWED_LABELS:
            matrix[expected][predicted] += 1
    return matrix


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
        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion.get(label, {}).values()),
        }
    return result


def _failure_modes(
    *,
    accuracy: float | None,
    macro_f1: float | None,
    min_accuracy: float,
    min_macro_f1: float,
    expected_counts: Mapping[str, int],
    predicted_counts: Mapping[str, int],
    metrics: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    modes: list[str] = []
    if accuracy is None or accuracy < min_accuracy:
        modes.append("accuracy_below_min")
    if macro_f1 is None or macro_f1 < min_macro_f1:
        modes.append("macro_f1_below_min")
    expected_labels = sorted(label for label in ALLOWED_LABELS if expected_counts.get(label, 0) > 0)
    for label in expected_labels:
        if int(predicted_counts.get(label, 0)) == 0:
            modes.append(f"missing_{label}_predictions")
        if metrics.get(label, {}).get("recall") == 0:
            modes.append(f"{label}_zero_recall")
    return modes


def _mismatch_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mismatches = [
        row
        for row in rows
        if row.get("expected_label") in ALLOWED_LABELS
        and row.get("policy_label") in ALLOWED_LABELS
        and row.get("expected_label") != row.get("policy_label")
    ]
    return {
        "schema_version": "prefix-local-judge-policy-mismatch-diagnostics-v1",
        "prompt_safe_summary": True,
        "mismatch_count": len(mismatches),
        "mismatch_label_transition_counts": _label_transition_counts(mismatches),
        "mismatch_semantic_hint_counts": _value_counts(mismatches, "semantic_hint"),
        "mismatch_risk_tag_counts": _risk_tag_counts(mismatches),
    }


def _policy_failure_diagnostics(policies: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for policy in policies:
        policy_name = str(policy.get("policy_name") or "")
        diagnostics = (
            policy.get("mismatch_diagnostics")
            if isinstance(policy.get("mismatch_diagnostics"), Mapping)
            else {}
        )
        if not policy_name:
            continue
        result[policy_name] = {
            "schema_version": "prefix-local-judge-policy-failure-diagnostics-v1",
            "prompt_safe_summary": True,
            "mismatch_count": _num(diagnostics.get("mismatch_count")),
            "top_mismatch_label_transitions": _top_counts(
                diagnostics.get("mismatch_label_transition_counts")
            ),
            "top_mismatch_semantic_hints": _top_counts(
                diagnostics.get("mismatch_semantic_hint_counts")
            ),
            "top_mismatch_risk_tags": _top_counts(diagnostics.get("mismatch_risk_tag_counts")),
        }
    return result


def _best_policy(policies: Sequence[Mapping[str, Any]], *, ready_only: bool) -> str | None:
    candidates = [policy for policy in policies if not ready_only or policy.get("ready") is True]
    if not candidates:
        return None
    best = sorted(
        candidates,
        key=lambda policy: (
            _score(policy.get("macro_f1")),
            _score(policy.get("accuracy")),
            str(policy.get("policy_name")),
        ),
        reverse=True,
    )[0]
    return str(best.get("policy_name"))


def _recommendation(*, best_ready: str | None, best_overall: str | None) -> str:
    if best_ready:
        return f"use_{best_ready}_for_goldset_scoped_validator_evidence"
    if best_overall:
        return f"no_policy_passed_thresholds_inspect_{best_overall}_and_add_better_rules_or_model"
    return "no_policy_evaluated_check_inputs"


def _safe_label(value: Any) -> str | None:
    label = str(value or "").strip().lower()
    return label if label in ALLOWED_LABELS else None


def _label_transition_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        expected = row.get("expected_label")
        predicted = row.get("policy_label")
        if expected in ALLOWED_LABELS and predicted in ALLOWED_LABELS:
            counts[f"{expected}->{predicted}"] += 1
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


def _risk_tag_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        values = row.get("risk_tags") if isinstance(row.get("risk_tags"), list) else []
        if not values:
            counts["none"] += 1
            continue
        for value in values:
            counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _top_counts(value: Any, *, limit: int = 5) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    items = sorted(
        ((str(key), _num(count)) for key, count in value.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return {key: count for key, count in items[:limit] if count > 0}


def _macro_value(metrics: Mapping[str, Mapping[str, Any]], field: str) -> float | None:
    values = [value.get(field) for value in metrics.values() if isinstance(value.get(field), (int, float))]
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _score(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else -1.0


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


def _write_json(path: str | Path | None, value: Mapping[str, Any]) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _write_text(path: str | Path | None, value: str) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value, encoding="utf-8")
    return target


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
