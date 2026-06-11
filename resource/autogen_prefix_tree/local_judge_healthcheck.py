from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .candidate_label_eval import (
    ALLOWED_LABELS,
    LocalModelJudgeConfig,
    _call_openai_compatible_judge,
    _parse_model_judge_result,
)


_SYNTHETIC_EXPECTED_LABEL = "accept"
_SYNTHETIC_CANDIDATE_TEXT = (
    "Stable shared verification rule for LOCAL_JUDGE_HEALTHCHECK_SYNTHETIC_TEXT: "
    "check evidence before giving the final answer."
)


@dataclass(frozen=True)
class LocalJudgeHealthcheckResult:
    summary_path: str | None
    summary: dict[str, Any]


def run_local_judge_healthcheck(
    *,
    base_url: str,
    model: str,
    api_key: str | None = None,
    timeout: float = 30.0,
    min_confidence: float = 0.66,
    summary_path: str | Path | None = None,
) -> LocalJudgeHealthcheckResult:
    """Call a local OpenAI-compatible candidate judge once and write a prompt-safe summary."""

    row = _synthetic_candidate_row()
    summary = _base_summary(
        base_url=base_url,
        model=model,
        api_key_configured=bool(api_key),
        timeout=timeout,
        min_confidence=min_confidence,
        row=row,
    )
    try:
        if not base_url.strip():
            raise ValueError("missing_base_url")
        if not model.strip():
            raise ValueError("missing_model")
        config = LocalModelJudgeConfig(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout,
        )
        summary["request_sent"] = True
        raw_result = _call_openai_compatible_judge(config, row)
        summary["response_parse_ok"] = True
        _apply_model_result(summary, raw_result, min_confidence=min_confidence)
    except Exception as exc:  # noqa: BLE001
        _set_error(summary, exc)

    summary["recommendation"] = _recommendation(summary)
    target = _write_json(summary_path, summary)
    return LocalJudgeHealthcheckResult(
        summary_path=str(target) if target else None,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Healthcheck a local OpenAI-compatible semantic candidate judge with one synthetic request."
    )
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:11434/v1.")
    parser.add_argument("--model", required=True, help="Local judge model name.")
    parser.add_argument(
        "--api-key-env",
        help="Optional environment variable containing the judge API key. The key is never written to output.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--min-confidence", type=float, default=0.66)
    parser.add_argument("--summary", help="Optional prompt-safe summary JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    result = run_local_judge_healthcheck(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        timeout=args.timeout,
        min_confidence=args.min_confidence,
        summary_path=args.summary,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0 if result.summary.get("ready") is True else 1


def _base_summary(
    *,
    base_url: str,
    model: str,
    api_key_configured: bool,
    timeout: float,
    min_confidence: float,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "prefix-local-judge-healthcheck-summary-v1",
        "prompt_safe_summary": True,
        "base_url": base_url,
        "model": model,
        "timeout_seconds": timeout,
        "min_confidence": min_confidence,
        "api_key_configured": api_key_configured,
        "request_sent": False,
        "response_parse_ok": False,
        "allowed_label": False,
        "label": None,
        "confidence": None,
        "confidence_meets_min": False,
        "synthetic_candidate": {
            "semantic_hint": row.get("semantic_hint"),
            "classifier_confidence": row.get("confidence"),
            "risk_tag_count": len(row.get("risk_tags") or ()),
            "char_count": row.get("char_count"),
            "line_count": row.get("line_count"),
            "expected_label": _SYNTHETIC_EXPECTED_LABEL,
        },
        "synthetic_label_matches_expected": False,
        "ready": False,
        "real_provider_metrics_available": False,
        "real_provider_metrics_note": (
            "This healthcheck only validates one local OpenAI-compatible judge call and JSON label parsing. "
            "It does not access provider APIs and cannot prove real cached tokens, latency, cost, task success, "
            "or broad semantic-model quality."
        ),
    }


def _apply_model_result(
    summary: dict[str, Any],
    raw_result: Mapping[str, Any],
    *,
    min_confidence: float,
) -> None:
    raw_label = str(raw_result.get("label") or "").strip().lower()
    summary["label"] = raw_label or None
    if raw_label not in ALLOWED_LABELS:
        summary["error_type"] = "ValueError"
        summary["error_reason"] = "invalid_model_label"
        return

    label, _reason, confidence = _parse_model_judge_result(raw_result)
    confidence_meets_min = confidence >= min_confidence
    summary["label"] = label
    summary["allowed_label"] = True
    summary["confidence"] = confidence
    summary["confidence_meets_min"] = confidence_meets_min
    summary["synthetic_label_matches_expected"] = label == _SYNTHETIC_EXPECTED_LABEL
    summary["ready"] = bool(summary["response_parse_ok"] and confidence_meets_min)


def _set_error(summary: dict[str, Any], exc: Exception) -> None:
    summary["error_type"] = type(exc).__name__
    summary["error_reason"] = _safe_error_reason(exc)


def _safe_error_reason(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return "response_content_not_json"
    if isinstance(exc, urllib.error.HTTPError):
        return f"http_status_{exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return "url_error"
    if isinstance(exc, KeyError):
        key = _safe_token(exc.args[0] if exc.args else "missing_key")
        return f"missing_response_field:{key}"
    if isinstance(exc, IndexError):
        return "missing_response_choice"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ValueError):
        return _safe_token(str(exc))
    return "local_judge_error"


def _recommendation(summary: Mapping[str, Any]) -> str:
    if not summary.get("ready"):
        return "fix_local_judge_endpoint_or_json_schema"
    if summary.get("synthetic_label_matches_expected"):
        return "run_small_local_judge_batch"
    return "inspect_synthetic_label_before_large_batch"


def _synthetic_candidate_row() -> dict[str, Any]:
    return {
        "schema_version": "prefix-semantic-candidate-v1",
        "candidate_id": "local-judge-healthcheck-synthetic-candidate",
        "source_path": "local_judge_healthcheck.synthetic",
        "symbol": "LOCAL_JUDGE_HEALTHCHECK",
        "parent_hash": "synthetic-parent",
        "text_hash": "synthetic-text",
        "semantic_hint": "verification_policy",
        "confidence": 0.91,
        "char_count": len(_SYNTHETIC_CANDIDATE_TEXT),
        "line_count": 1,
        "risk_tags": [],
        "text": _SYNTHETIC_CANDIDATE_TEXT,
    }


def _write_json(path: str | Path | None, value: Mapping[str, Any]) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _safe_token(value: Any) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(value).strip().lower()).strip("_")
    return cleaned[:80] or "local_judge_error"


if __name__ == "__main__":
    raise SystemExit(main())
