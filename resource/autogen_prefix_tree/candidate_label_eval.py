from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ACCEPTED_HINTS = {"tool_or_code_policy", "verification_policy", "team_policy", "procedure_policy"}
HIGH_RISK_TAGS = {"agent_identity_boundary", "possible_turn_specific_reference", "user_interaction_constraint"}
ALLOWED_LABELS = {"accept", "review", "reject"}
RULE_JUDGE = "rule"
OPENAI_COMPATIBLE_JUDGE = "openai-compatible"
LOCAL_JUDGE_SCOPE_ALL = "all"
LOCAL_JUDGE_SCOPE_REVIEW = "review"


@dataclass(frozen=True)
class CandidateLabelEvalResult:
    input_path: str
    output_path: str | None
    summary_path: str | None
    summary: dict[str, Any]


@dataclass(frozen=True)
class LocalModelJudgeConfig:
    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 30.0


def evaluate_candidate_labels(
    *,
    input_path: str | Path,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    min_confidence: float = 0.66,
    judge: str = RULE_JUDGE,
    local_judge_base_url: str | None = None,
    local_judge_model: str | None = None,
    local_judge_api_key: str | None = None,
    local_judge_timeout: float = 30.0,
    local_judge_scope: str = LOCAL_JUDGE_SCOPE_ALL,
    max_local_judge_calls: int | None = None,
) -> CandidateLabelEvalResult:
    source_path = Path(input_path)
    rows = tuple(_load_jsonl(source_path))
    judge_mode = _normalize_judge_mode(judge)
    scope = _normalize_local_judge_scope(local_judge_scope)
    local_model_config = _build_local_model_config(
        judge_mode=judge_mode,
        base_url=local_judge_base_url,
        model=local_judge_model,
        api_key=local_judge_api_key,
        timeout_seconds=local_judge_timeout,
    )
    normalized_max_local_judge_calls = _normalize_optional_nonnegative_int(max_local_judge_calls)
    rule_labeled_rows = tuple(_label_row_by_rules(row, min_confidence=min_confidence) for row in rows)
    selected_for_local_judge = _selected_local_judge_row_indexes(
        rule_labeled_rows,
        judge_mode=judge_mode,
        local_judge_scope=scope,
        max_local_judge_calls=normalized_max_local_judge_calls,
    )
    labeled: list[dict[str, Any]] = []
    for row_index, (row, rule_labeled) in enumerate(zip(rows, rule_labeled_rows)):
        labeled_row = _label_row(
            row,
            row_index=row_index,
            rule_labeled=rule_labeled,
            judge_mode=judge_mode,
            local_model_config=local_model_config,
            local_judge_scope=scope,
            selected_for_local_judge=selected_for_local_judge,
            min_confidence=min_confidence,
        )
        labeled.append(labeled_row)
    labeled_rows = tuple(labeled)
    summary = _summarize(
        labeled_rows,
        input_path=str(source_path),
        min_confidence=min_confidence,
        judge_mode=judge_mode,
        local_model_config=local_model_config,
        local_judge_scope=scope,
        max_local_judge_calls=max_local_judge_calls,
    )
    output_target = _write_jsonl(output_path, labeled_rows)
    summary_target = _write_json(summary_path, summary)
    return CandidateLabelEvalResult(
        input_path=str(source_path),
        output_path=str(output_target) if output_target else None,
        summary_path=str(summary_target) if summary_target else None,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic candidate labels with a conservative rule judge or optional local model judge."
    )
    parser.add_argument("--input", required=True, help="Input semantic candidate JSONL.")
    parser.add_argument("--output", help="Optional labeled candidate JSONL output.")
    parser.add_argument("--summary", help="Optional label summary JSON output.")
    parser.add_argument("--min-confidence", type=float, default=0.66)
    parser.add_argument("--judge", choices=(RULE_JUDGE, OPENAI_COMPATIBLE_JUDGE), default=RULE_JUDGE)
    parser.add_argument(
        "--local-judge-base-url",
        help="OpenAI-compatible base URL for --judge openai-compatible, e.g. http://127.0.0.1:11434/v1.",
    )
    parser.add_argument("--local-judge-model", help="Model name for --judge openai-compatible.")
    parser.add_argument(
        "--local-judge-api-key-env",
        help="Optional environment variable containing the local judge API key. The key is never written to output.",
    )
    parser.add_argument("--local-judge-timeout", type=float, default=30.0)
    parser.add_argument(
        "--local-judge-scope",
        choices=(LOCAL_JUDGE_SCOPE_ALL, LOCAL_JUDGE_SCOPE_REVIEW),
        default=LOCAL_JUDGE_SCOPE_ALL,
        help="For --judge openai-compatible, choose whether to judge all candidates or only rule-review rows.",
    )
    parser.add_argument(
        "--max-local-judge-calls",
        type=int,
        help="Optional cap on actual local model calls for --judge openai-compatible.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = os.environ.get(args.local_judge_api_key_env) if args.local_judge_api_key_env else None
    result = evaluate_candidate_labels(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary,
        min_confidence=args.min_confidence,
        judge=args.judge,
        local_judge_base_url=args.local_judge_base_url,
        local_judge_model=args.local_judge_model,
        local_judge_api_key=api_key,
        local_judge_timeout=args.local_judge_timeout,
        local_judge_scope=args.local_judge_scope,
        max_local_judge_calls=args.max_local_judge_calls,
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


def _label_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    rule_labeled: Mapping[str, Any],
    judge_mode: str,
    local_model_config: LocalModelJudgeConfig | None,
    local_judge_scope: str,
    selected_for_local_judge: set[int] | None,
    min_confidence: float,
) -> dict[str, Any]:
    if judge_mode == RULE_JUDGE:
        return dict(rule_labeled)
    assert local_model_config is not None
    if local_judge_scope == LOCAL_JUDGE_SCOPE_REVIEW and rule_labeled.get("label") != "review":
        return _label_row_skipped_by_local_judge_scope(rule_labeled)
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return _label_row_by_local_model(
            row,
            rule_labeled=rule_labeled,
            min_confidence=min_confidence,
            config=local_model_config,
        )
    if selected_for_local_judge is not None and row_index not in selected_for_local_judge:
        return _label_row_skipped_by_local_judge_budget(rule_labeled)
    return _label_row_by_local_model(
        row,
        rule_labeled=rule_labeled,
        min_confidence=min_confidence,
        config=local_model_config,
    )


def _label_row_by_rules(row: Mapping[str, Any], *, min_confidence: float) -> dict[str, Any]:
    semantic_hint = str(row.get("semantic_hint") or "")
    confidence = _float(row.get("confidence"))
    risk_tags = tuple(str(tag) for tag in row.get("risk_tags") or ())
    high_risk = sorted(set(risk_tags) & HIGH_RISK_TAGS)
    if semantic_hint not in ACCEPTED_HINTS:
        label = "reject"
        reason = "unsupported_semantic_hint"
    elif confidence < min_confidence:
        label = "review"
        reason = "low_confidence"
    elif high_risk:
        label = "review"
        reason = "high_risk_boundary:" + ",".join(high_risk)
    else:
        label = "accept"
        reason = "accepted_by_conservative_rule_judge"
    labeled = dict(row)
    labeled.update(
        {
            "label": label,
            "label_source": "conservative_rule_judge_v1",
            "label_reason": reason,
            "label_notes": None,
            "rule_label": label,
            "rule_label_reason": reason,
        }
    )
    return labeled


def _selected_local_judge_row_indexes(
    rows: Sequence[Mapping[str, Any]],
    *,
    judge_mode: str,
    local_judge_scope: str,
    max_local_judge_calls: int | None,
) -> set[int] | None:
    if judge_mode == RULE_JUDGE or max_local_judge_calls is None:
        return None
    eligible = [
        (index, row)
        for index, row in enumerate(rows)
        if _is_local_judge_budget_candidate(row, local_judge_scope=local_judge_scope)
    ]
    selected = sorted(eligible, key=lambda item: _local_judge_budget_sort_key(item[1], item[0]))[
        :max_local_judge_calls
    ]
    return {index for index, _row in selected}


def _is_local_judge_budget_candidate(row: Mapping[str, Any], *, local_judge_scope: str) -> bool:
    if local_judge_scope == LOCAL_JUDGE_SCOPE_REVIEW and row.get("label") != "review":
        return False
    text = row.get("text")
    return isinstance(text, str) and bool(text.strip())


def _local_judge_budget_sort_key(
    row: Mapping[str, Any],
    row_index: int,
) -> tuple[int, int, float, int, str, str, int]:
    risk_tags = {str(tag) for tag in row.get("risk_tags") or ()}
    high_risk = len(risk_tags & HIGH_RISK_TAGS)
    label = str(row.get("label") or "")
    confidence = _float(row.get("confidence"))
    return (
        0 if label == "review" else 1,
        -high_risk,
        confidence,
        -_int(row.get("char_count")),
        str(row.get("source_path") or ""),
        str(row.get("candidate_id") or ""),
        row_index,
    )


def _label_row_by_local_model(
    row: Mapping[str, Any],
    *,
    rule_labeled: Mapping[str, Any],
    min_confidence: float,
    config: LocalModelJudgeConfig,
) -> dict[str, Any]:
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        labeled = dict(row)
        labeled.update(
            {
                "label": "review",
                "label_source": "openai_compatible_local_judge_v1",
                "label_reason": "missing_candidate_text",
                "label_notes": "Rerun candidate export with --include-candidate-text before using the local model judge.",
                "rule_label": rule_labeled.get("label"),
                "rule_label_reason": rule_labeled.get("label_reason"),
                "local_judge_action": "missing_candidate_text",
            }
        )
        return labeled
    try:
        model_result = _call_openai_compatible_judge(config, row)
        model_label, model_reason, model_confidence = _parse_model_judge_result(model_result)
    except Exception as exc:
        labeled = dict(row)
        labeled.update(
            {
                "label": "review",
                "label_source": "openai_compatible_local_judge_v1",
                "label_reason": f"local_judge_error:{type(exc).__name__}",
                "label_notes": None,
                "rule_label": rule_labeled.get("label"),
                "rule_label_reason": rule_labeled.get("label_reason"),
                "local_judge_action": "local_judge_error",
            }
        )
        return labeled

    final_label, final_reason = _apply_static_safety_clamp(
        row,
        model_label=model_label,
        model_reason=model_reason,
        model_confidence=model_confidence,
        min_confidence=min_confidence,
    )
    labeled = dict(row)
    labeled.update(
        {
            "label": final_label,
            "label_source": "openai_compatible_local_judge_v1",
            "label_reason": final_reason,
            "label_notes": None,
            "model_label": model_label,
            "model_confidence": model_confidence,
            "rule_label": rule_labeled.get("label"),
            "rule_label_reason": rule_labeled.get("label_reason"),
            "local_judge_action": "model_called",
        }
    )
    return labeled


def _label_row_skipped_by_local_judge_scope(rule_labeled: Mapping[str, Any]) -> dict[str, Any]:
    labeled = dict(rule_labeled)
    labeled.update(
        {
            "rule_label": rule_labeled.get("label"),
            "rule_label_reason": rule_labeled.get("label_reason"),
            "local_judge_action": "skipped_by_scope",
        }
    )
    return labeled


def _label_row_skipped_by_local_judge_budget(rule_labeled: Mapping[str, Any]) -> dict[str, Any]:
    labeled = dict(rule_labeled)
    labeled.update(
        {
            "rule_label": rule_labeled.get("label"),
            "rule_label_reason": rule_labeled.get("label_reason"),
            "local_judge_action": "skipped_by_budget",
        }
    )
    return labeled


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    input_path: str,
    min_confidence: float,
    judge_mode: str,
    local_model_config: LocalModelJudgeConfig | None,
    local_judge_scope: str,
    max_local_judge_calls: int | None,
) -> dict[str, Any]:
    label_counts = Counter(str(row.get("label") or "missing") for row in rows)
    hint_counts = Counter(str(row.get("semantic_hint") or "unknown") for row in rows)
    accepted_hint_counts = Counter(
        str(row.get("semantic_hint") or "unknown") for row in rows if row.get("label") == "accept"
    )
    reason_counts = Counter(str(row.get("label_reason") or "missing") for row in rows)
    rule_label_counts = Counter(str(row.get("rule_label") or "missing") for row in rows if "rule_label" in row)
    model_label_counts = Counter(str(row.get("model_label") or "missing") for row in rows if "model_label" in row)
    rule_to_model_counts = Counter(
        _transition_key(row.get("rule_label"), row.get("model_label")) for row in rows if "model_label" in row
    )
    model_to_final_counts = Counter(
        _transition_key(row.get("model_label"), row.get("label")) for row in rows if "model_label" in row
    )
    rule_to_final_counts = Counter(
        _transition_key(row.get("rule_label"), row.get("label")) for row in rows if "rule_label" in row
    )
    static_clamp_count = sum(
        1 for row in rows if str(row.get("label_reason") or "").startswith("model_accept_clamped_")
    )
    local_judge_action_counts = Counter(
        str(row.get("local_judge_action") or "missing") for row in rows if "local_judge_action" in row
    )
    local_judge_skipped_by_scope_label_counts = Counter(
        str(row.get("rule_label") or "missing")
        for row in rows
        if row.get("local_judge_action") == "skipped_by_scope"
    )
    local_judge_skipped_by_budget_label_counts = Counter(
        str(row.get("rule_label") or "missing")
        for row in rows
        if row.get("local_judge_action") == "skipped_by_budget"
    )
    local_judge_effectiveness = _local_judge_effectiveness_diagnostics(rows)
    review_queue = _review_queue_diagnostics(rows)
    summary: dict[str, Any] = {
        "schema_version": "prefix-semantic-candidate-label-summary-v1",
        "input_path": input_path,
        "candidate_count": len(rows),
        "min_confidence": min_confidence,
        "judge_mode": judge_mode,
        "local_judge_scope": local_judge_scope,
        "max_local_judge_calls": max_local_judge_calls,
        "local_judge_budget_strategy": "priority_review_risk_confidence_chars_v1"
        if max_local_judge_calls is not None
        else None,
        "local_judge": _local_judge_summary(local_model_config),
        "label_counts": dict(sorted(label_counts.items())),
        "semantic_hint_counts": dict(sorted(hint_counts.items())),
        "accepted_semantic_hint_counts": dict(sorted(accepted_hint_counts.items())),
        "label_reason_counts": dict(sorted(reason_counts.items())),
        "reviewed_risk_tag_counts": review_queue["risk_tag_counts"],
        "review_queue_diagnostics": review_queue,
        "static_safety_clamp_count": static_clamp_count,
    }
    if rule_label_counts:
        summary["rule_label_counts"] = dict(sorted(rule_label_counts.items()))
    if model_label_counts:
        summary["model_label_counts"] = dict(sorted(model_label_counts.items()))
    if rule_to_model_counts:
        summary["rule_to_model_label_counts"] = dict(sorted(rule_to_model_counts.items()))
    if model_to_final_counts:
        summary["model_to_final_label_counts"] = dict(sorted(model_to_final_counts.items()))
    if rule_to_final_counts:
        summary["rule_to_final_label_counts"] = dict(sorted(rule_to_final_counts.items()))
    if local_judge_action_counts:
        summary["local_judge_action_counts"] = dict(sorted(local_judge_action_counts.items()))
    if local_judge_skipped_by_scope_label_counts:
        summary["local_judge_skipped_by_scope_label_counts"] = dict(
            sorted(local_judge_skipped_by_scope_label_counts.items())
        )
    if local_judge_skipped_by_budget_label_counts:
        summary["local_judge_skipped_by_budget_label_counts"] = dict(
            sorted(local_judge_skipped_by_budget_label_counts.items())
        )
    if judge_mode == OPENAI_COMPATIBLE_JUDGE:
        summary["local_judge_budget_diagnostics"] = _local_judge_budget_diagnostics(
            max_local_judge_calls=max_local_judge_calls,
            action_counts=local_judge_action_counts,
        )
        summary["local_judge_effectiveness_diagnostics"] = local_judge_effectiveness
    return summary


def _local_judge_effectiveness_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rule_review_rows = [row for row in rows if row.get("rule_label") == "review"]
    called_rule_review_rows = [
        row for row in rule_review_rows if row.get("local_judge_action") == "model_called"
    ]
    final_label_counts = Counter(str(row.get("label") or "missing") for row in rule_review_rows)
    model_label_counts = Counter(
        str(row.get("model_label") or "missing") for row in called_rule_review_rows
    )
    action_counts = Counter(str(row.get("local_judge_action") or "missing") for row in rule_review_rows)
    resolved_rows = [
        row for row in rule_review_rows if row.get("label") in {"accept", "reject"}
    ]
    called_resolved_rows = [
        row for row in called_rule_review_rows if row.get("label") in {"accept", "reject"}
    ]
    return {
        "schema_version": "prefix-local-judge-effectiveness-diagnostics-v1",
        "rule_review_candidate_count": len(rule_review_rows),
        "model_called_rule_review_count": len(called_rule_review_rows),
        "resolved_rule_review_count": len(resolved_rows),
        "called_resolved_rule_review_count": len(called_resolved_rows),
        "remaining_rule_review_count": sum(1 for row in rule_review_rows if row.get("label") == "review"),
        "resolution_rate": _ratio_or_none(len(resolved_rows), len(rule_review_rows)),
        "called_resolution_rate": _ratio_or_none(len(called_resolved_rows), len(called_rule_review_rows)),
        "rule_review_final_label_counts": dict(sorted(final_label_counts.items())),
        "rule_review_model_label_counts": dict(sorted(model_label_counts.items())),
        "rule_review_local_judge_action_counts": dict(sorted(action_counts.items())),
    }


def _local_judge_budget_diagnostics(
    *,
    max_local_judge_calls: int | None,
    action_counts: Mapping[str, int],
) -> dict[str, Any]:
    model_called = _int(action_counts.get("model_called"))
    skipped_by_budget = _int(action_counts.get("skipped_by_budget"))
    eligible_for_budget = model_called + skipped_by_budget
    budget_limited = max_local_judge_calls is not None
    budget_exhausted = bool(budget_limited and skipped_by_budget > 0)
    return {
        "schema_version": "prefix-local-judge-budget-diagnostics-v1",
        "budget_limited": budget_limited,
        "max_local_judge_calls": max_local_judge_calls,
        "budget_strategy": "priority_review_risk_confidence_chars_v1" if budget_limited else None,
        "eligible_for_budget_count": eligible_for_budget,
        "model_called_count": model_called,
        "skipped_by_budget_count": skipped_by_budget,
        "budget_exhausted": budget_exhausted,
        "budget_coverage_rate": _ratio_or_none(model_called, eligible_for_budget),
    }


def _review_queue_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reviewed = [row for row in rows if row.get("label") == "review"]
    semantic_hint_counts = Counter(str(row.get("semantic_hint") or "unknown") for row in reviewed)
    reason_counts = Counter(str(row.get("label_reason") or "missing") for row in reviewed)
    risk_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    parent_hashes: set[str] = set()
    total_chars = 0
    for row in reviewed:
        risk_counts.update(str(tag) for tag in row.get("risk_tags") or ())
        source = row.get("source_path")
        if source:
            source_counts[str(source)] += 1
        parent_hash = row.get("parent_hash")
        if parent_hash:
            parent_hashes.add(str(parent_hash))
        total_chars += _int(row.get("char_count"))
    top_sources = [
        {
            "source_path": source_path,
            "review_candidate_count": count,
        }
        for source_path, count in source_counts.most_common(10)
    ]
    return {
        "schema_version": "prefix-review-queue-diagnostics-v1",
        "review_candidate_count": len(reviewed),
        "review_parent_block_count": len(parent_hashes),
        "review_source_file_count": len(source_counts),
        "review_candidate_chars": total_chars,
        "semantic_hint_counts": dict(sorted(semantic_hint_counts.items())),
        "risk_tag_counts": dict(sorted(risk_counts.items())),
        "label_reason_counts": dict(sorted(reason_counts.items())),
        "top_source_files": top_sources,
        "local_judge_priority": _local_judge_priority(
            review_count=len(reviewed),
            risk_counts=risk_counts,
            reason_counts=reason_counts,
        ),
    }


def _call_openai_compatible_judge(config: LocalModelJudgeConfig, row: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "model": config.model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You label semantic prompt candidates for a prefix-reordering cache experiment. "
                    "Return one JSON object only with keys label, reason, and confidence. "
                    "Do not include markdown, prose, comments, code fences, or repeated objects. "
                    "Allowed label values are accept, review, reject. "
                    "accept means the text is a stable shared policy/procedure/tool/verification/team instruction "
                    "that could be considered for a shared prefix after normal validator checks. "
                    "review means the candidate may be useful but has identity, privacy, turn-specific, or conditional risk. "
                    "reject means it is not a stable shared prefix candidate."
                ),
            },
            {"role": "user", "content": json.dumps(_local_model_candidate_payload(row), ensure_ascii=False)},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    request = urllib.request.Request(
        _chat_completions_url(config.base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "\n".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
    if not isinstance(content, str):
        raise ValueError("missing_model_content")
    return _parse_json_object(content)


def _local_model_candidate_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": row.get("candidate_id"),
        "semantic_hint": row.get("semantic_hint"),
        "classifier_confidence": row.get("confidence"),
        "risk_tags": row.get("risk_tags") or [],
        "char_count": row.get("char_count"),
        "line_count": row.get("line_count"),
        "text": row.get("text") or "",
    }


def _parse_model_judge_result(value: Mapping[str, Any]) -> tuple[str, str, float]:
    label = str(value.get("label") or "").strip().lower()
    if label not in ALLOWED_LABELS:
        raise ValueError("invalid_model_label")
    reason = _safe_reason(value.get("reason") or "model_judged")
    confidence = _float(value.get("confidence"))
    return label, reason, confidence


def _apply_static_safety_clamp(
    row: Mapping[str, Any],
    *,
    model_label: str,
    model_reason: str,
    model_confidence: float,
    min_confidence: float,
) -> tuple[str, str]:
    if model_label != "accept":
        return model_label, f"local_model:{model_reason}"
    if model_confidence < min_confidence:
        return "review", f"model_accept_clamped_low_confidence:{model_confidence:.3g}"
    semantic_hint = str(row.get("semantic_hint") or "")
    risk_tags = tuple(str(tag) for tag in row.get("risk_tags") or ())
    high_risk = sorted(set(risk_tags) & HIGH_RISK_TAGS)
    if semantic_hint not in ACCEPTED_HINTS:
        return "review", f"model_accept_clamped_unsupported_hint:{_safe_reason(semantic_hint)}"
    if high_risk:
        return "review", "model_accept_clamped_by_static_safety:" + ",".join(high_risk)
    return "accept", f"local_model:{model_reason}"


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            value = json.loads(stripped[start : end + 1])
        elif _looks_like_structured_label_text(stripped):
            value = _parse_structured_label_text(stripped)
        else:
            raise exc
    if not isinstance(value, dict):
        raise ValueError("model_response_not_object")
    return value


def _parse_structured_label_text(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized_key not in {"label", "reason", "confidence"}:
            continue
        parsed[normalized_key] = raw_value.strip().strip("`\"'")
    label = str(parsed.get("label") or "").strip().lower().rstrip(".,;")
    if label not in ALLOWED_LABELS:
        raise ValueError("invalid_structured_label_text")
    confidence = _structured_confidence(parsed.get("confidence"))
    if confidence is None:
        confidence = 0.0 if label == "review" else 0.5
    return {
        "label": label,
        "reason": parsed.get("reason") or "structured_text_fallback",
        "confidence": confidence,
    }


def _looks_like_structured_label_text(text: str) -> bool:
    return any(line.strip().lower().startswith("label:") for line in text.splitlines())


def _structured_confidence(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    if not match:
        return None
    confidence = float(match.group(0))
    if confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0
    return max(0.0, min(1.0, confidence))


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _normalize_judge_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {RULE_JUDGE, OPENAI_COMPATIBLE_JUDGE}:
        raise ValueError(f"Unsupported judge mode: {value}")
    return normalized


def _normalize_local_judge_scope(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    if normalized == "review-only":
        normalized = LOCAL_JUDGE_SCOPE_REVIEW
    if normalized not in {LOCAL_JUDGE_SCOPE_ALL, LOCAL_JUDGE_SCOPE_REVIEW}:
        raise ValueError(f"Unsupported local judge scope: {value}")
    return normalized


def _normalize_optional_nonnegative_int(value: int | None) -> int | None:
    if value is None:
        return None
    if value < 0:
        raise ValueError("max_local_judge_calls must be non-negative")
    return value


def _build_local_model_config(
    *,
    judge_mode: str,
    base_url: str | None,
    model: str | None,
    api_key: str | None,
    timeout_seconds: float,
) -> LocalModelJudgeConfig | None:
    if judge_mode == RULE_JUDGE:
        return None
    if not base_url:
        raise ValueError("--local-judge-base-url is required for --judge openai-compatible")
    if not model:
        raise ValueError("--local-judge-model is required for --judge openai-compatible")
    return LocalModelJudgeConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )


def _local_judge_summary(config: LocalModelJudgeConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "base_url": config.base_url,
        "model": config.model,
        "timeout_seconds": config.timeout_seconds,
    }


def _safe_reason(value: Any) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(value).strip().lower()).strip("_")
    return cleaned[:80] or "model_judged"


def _transition_key(before: Any, after: Any) -> str:
    return f"{_safe_label(before)}->{_safe_label(after)}"


def _safe_label(value: Any) -> str:
    label = str(value or "missing").strip().lower()
    return label if label in ALLOWED_LABELS else "missing"


def _local_judge_priority(
    *,
    review_count: int,
    risk_counts: Mapping[str, int],
    reason_counts: Mapping[str, int],
) -> str:
    if review_count == 0:
        return "none"
    if _int(reason_counts.get("missing_candidate_text")) > 0:
        return "rerun_with_candidate_text_before_local_judge"
    high_risk_count = sum(_int(risk_counts.get(tag)) for tag in HIGH_RISK_TAGS)
    if high_risk_count > 0:
        return "high_risk_review_with_local_judge_and_static_clamps"
    return "medium_priority_local_judge_review"


def _write_jsonl(path: str | Path | None, rows: Sequence[Mapping[str, Any]]) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def _write_json(path: str | Path | None, value: Mapping[str, Any]) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


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


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


if __name__ == "__main__":
    raise SystemExit(main())
