from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .candidate_label_eval import evaluate_candidate_labels
from .candidate_utility_eval import evaluate_candidate_utility
from .review_worklist import build_review_worklist
from .source_prompt_corpus import evaluate_source_prompt_corpus


@dataclass(frozen=True)
class OfflineSemanticPipelineResult:
    output_dir: str
    summary_path: str
    summary: dict[str, Any]


def run_offline_semantic_pipeline(
    *,
    source_root: str | Path,
    output_dir: str | Path,
    session_id: str = "offline-semantic-pipeline",
    include_tests: bool = False,
    include_candidate_text: bool = False,
    min_prompt_chars: int = 40,
    min_confidence: float = 0.66,
    enable_natural_language_segmentation: bool = False,
    judge: str = "rule",
    local_judge_base_url: str | None = None,
    local_judge_model: str | None = None,
    local_judge_api_key: str | None = None,
    local_judge_timeout: float = 30.0,
    local_judge_scope: str = "all",
    max_local_judge_calls: int | None = None,
) -> OfflineSemanticPipelineResult:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    requests_path = out / "source_prompts.jsonl"
    dataset_summary_path = out / "dataset_summary.json"
    telemetry_path = out / "dataset_telemetry.jsonl"
    candidates_path = out / "semantic_candidates.jsonl"
    labeled_candidates_path = out / "semantic_candidates_labeled.jsonl"
    label_summary_path = out / "candidate_label_summary.json"
    review_worklist_path = out / "review_worklist.jsonl"
    review_worklist_summary_path = out / "review_worklist_summary.json"
    review_worklist_with_text_path = out / "review_worklist_with_text.jsonl"
    review_worklist_with_text_summary_path = out / "review_worklist_with_text_summary.json"
    utility_summary_path = out / "candidate_utility_summary.json"
    utility_with_review_summary_path = out / "candidate_utility_summary_with_review.json"
    pipeline_summary_path = out / "pipeline_summary.json"

    corpus_result = evaluate_source_prompt_corpus(
        source_root=source_root,
        output_requests_path=requests_path,
        summary_path=dataset_summary_path,
        telemetry_path=telemetry_path,
        candidates_path=candidates_path,
        session_id=session_id,
        include_tests=include_tests,
        include_candidate_text=include_candidate_text,
        min_prompt_chars=min_prompt_chars,
        enable_natural_language_segmentation=enable_natural_language_segmentation,
    )
    label_result = evaluate_candidate_labels(
        input_path=candidates_path,
        output_path=labeled_candidates_path,
        summary_path=label_summary_path,
        min_confidence=min_confidence,
        judge=judge,
        local_judge_base_url=local_judge_base_url,
        local_judge_model=local_judge_model,
        local_judge_api_key=local_judge_api_key,
        local_judge_timeout=local_judge_timeout,
        local_judge_scope=local_judge_scope,
        max_local_judge_calls=max_local_judge_calls,
    )
    review_worklist_result = build_review_worklist(
        input_path=labeled_candidates_path,
        output_path=review_worklist_path,
        summary_path=review_worklist_summary_path,
    )
    review_text_worklist_result = (
        build_review_worklist(
            input_path=labeled_candidates_path,
            output_path=review_worklist_with_text_path,
            summary_path=review_worklist_with_text_summary_path,
            include_text=True,
        )
        if include_candidate_text
        else None
    )
    utility_result = evaluate_candidate_utility(
        input_path=labeled_candidates_path,
        summary_path=utility_summary_path,
    )
    utility_with_review_result = evaluate_candidate_utility(
        input_path=labeled_candidates_path,
        summary_path=utility_with_review_summary_path,
        include_review=True,
    )
    semantic_rule_gap_summary = _semantic_rule_gap_summary(
        dataset_summary=corpus_result.summary,
        label_summary=label_result.summary,
        utility_summary=utility_result.summary,
        utility_with_review_summary=utility_with_review_result.summary,
    )
    summary = {
        "schema_version": "prefix-offline-semantic-pipeline-summary-v1",
        "source_root": str(source_root),
        "output_dir": str(out),
        "session_id": session_id,
        "natural_language_segmentation_enabled": enable_natural_language_segmentation,
        "artifacts": {
            "requests": str(requests_path),
            "dataset_summary": str(dataset_summary_path),
            "dataset_telemetry": str(telemetry_path),
            "semantic_candidates": str(candidates_path),
            "semantic_candidates_labeled": str(labeled_candidates_path),
            "candidate_label_summary": str(label_summary_path),
            "review_worklist": str(review_worklist_path),
            "review_worklist_summary": str(review_worklist_summary_path),
            "candidate_utility_summary": str(utility_summary_path),
            "candidate_utility_summary_with_review": str(utility_with_review_summary_path),
        },
        "source_prompt_corpus": {
            "prompt_count": corpus_result.prompt_count,
            "source_file_count": corpus_result.source_file_count,
            "extraction_counts": corpus_result.extraction_counts,
        },
        "dataset_eval": {
            "supported_request_count": corpus_result.summary.get("supported_request_count"),
            "applied_count": corpus_result.summary.get("applied_count"),
            "reusable_prefix_request_count": corpus_result.summary.get("reusable_prefix_request_count"),
            "moved_semantic_type_counts": corpus_result.summary.get("moved_semantic_type_counts"),
            "validation_reason_counts": corpus_result.summary.get("validation_reason_counts"),
            "total_estimated_gain_chars": corpus_result.summary.get("total_estimated_gain_chars"),
            "semantic_type_counts": corpus_result.summary.get("semantic_type_counts"),
            "rule_gap_diagnostics": corpus_result.summary.get("rule_gap_diagnostics"),
        },
        "candidate_label_eval": label_result.summary,
        "review_worklist": review_worklist_result.summary,
        "review_worklist_with_text": review_text_worklist_result.summary if review_text_worklist_result else None,
        "candidate_utility_eval": utility_result.summary,
        "candidate_utility_eval_with_review": utility_with_review_result.summary,
        "semantic_rule_gap_summary": semantic_rule_gap_summary,
        "real_provider_metrics_available": False,
        "real_provider_metrics_note": "Offline character-level proxy only; no provider cached tokens, latency, cost, or task success metrics.",
    }
    if review_text_worklist_result is not None:
        summary["artifacts"]["review_worklist_with_text"] = str(review_worklist_with_text_path)
        summary["artifacts"]["review_worklist_with_text_summary"] = str(review_worklist_with_text_summary_path)
    pipeline_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return OfflineSemanticPipelineResult(
        output_dir=str(out),
        summary_path=str(pipeline_summary_path),
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run source prompt coverage, semantic candidate labeling, and utility proxy evaluation."
    )
    parser.add_argument("--source-root", required=True, help="Local source tree to scan.")
    parser.add_argument("--output-dir", required=True, help="Directory for all pipeline artifacts.")
    parser.add_argument("--session-id", default="offline-semantic-pipeline")
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument("--include-candidate-text", action="store_true")
    parser.add_argument("--min-prompt-chars", type=int, default=40)
    parser.add_argument("--min-confidence", type=float, default=0.66)
    parser.add_argument(
        "--enable-natural-language-segmentation",
        action="store_true",
        help="Opt-in offline analysis mode for unmarked natural-language system prompts.",
    )
    parser.add_argument("--judge", choices=("rule", "openai-compatible"), default="rule")
    parser.add_argument("--local-judge-base-url")
    parser.add_argument("--local-judge-model")
    parser.add_argument("--local-judge-api-key-env")
    parser.add_argument("--local-judge-timeout", type=float, default=30.0)
    parser.add_argument(
        "--local-judge-scope",
        choices=("all", "review"),
        default="all",
        help="For --judge openai-compatible, call the local judge for all candidates or only rule-review candidates.",
    )
    parser.add_argument(
        "--max-local-judge-calls",
        type=int,
        help="Optional cap on actual local model calls for --judge openai-compatible.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_offline_semantic_pipeline(
        source_root=args.source_root,
        output_dir=args.output_dir,
        session_id=args.session_id,
        include_tests=args.include_tests,
        include_candidate_text=args.include_candidate_text,
        min_prompt_chars=args.min_prompt_chars,
        min_confidence=args.min_confidence,
        enable_natural_language_segmentation=args.enable_natural_language_segmentation,
        judge=args.judge,
        local_judge_base_url=args.local_judge_base_url,
        local_judge_model=args.local_judge_model,
        local_judge_api_key=os.environ.get(args.local_judge_api_key_env) if args.local_judge_api_key_env else None,
        local_judge_timeout=args.local_judge_timeout,
        local_judge_scope=args.local_judge_scope,
        max_local_judge_calls=args.max_local_judge_calls,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


def _semantic_rule_gap_summary(
    *,
    dataset_summary: dict[str, Any],
    label_summary: dict[str, Any],
    utility_summary: dict[str, Any],
    utility_with_review_summary: dict[str, Any],
) -> dict[str, Any]:
    supported_request_count = _int(dataset_summary.get("supported_request_count"))
    applied_count = _int(dataset_summary.get("applied_count"))
    reusable_prefix_request_count = _int(dataset_summary.get("reusable_prefix_request_count"))
    accepted_candidate_count = _int((label_summary.get("label_counts") or {}).get("accept"))
    review_candidate_count = _int((label_summary.get("label_counts") or {}).get("review"))
    eligible_parent_block_count = _int(utility_summary.get("eligible_parent_block_count"))
    eligible_with_review_parent_block_count = _int(utility_with_review_summary.get("eligible_parent_block_count"))
    return {
        "schema_version": "prefix-semantic-rule-gap-summary-v1",
        "supported_request_count": supported_request_count,
        "applied_count": applied_count,
        "applied_rate": _ratio_or_none(applied_count, supported_request_count),
        "reusable_prefix_request_count": reusable_prefix_request_count,
        "candidate_count": _int(label_summary.get("candidate_count")),
        "accepted_candidate_count": accepted_candidate_count,
        "review_candidate_count": review_candidate_count,
        "accepted_candidate_parent_block_count": eligible_parent_block_count,
        "accepted_or_review_candidate_parent_block_count": eligible_with_review_parent_block_count,
        "accepted_repeated_group_count": _int(utility_summary.get("repeated_candidate_group_count")),
        "accepted_estimated_reusable_chars": _int(utility_summary.get("total_estimated_reusable_chars")),
        "accepted_or_review_estimated_reusable_chars": _int(
            utility_with_review_summary.get("total_estimated_reusable_chars")
        ),
        "candidate_promotion_policy": utility_summary.get("promotion_policy")
        if isinstance(utility_summary.get("promotion_policy"), dict)
        else {},
        "candidate_review_upper_bound_policy": utility_with_review_summary.get("promotion_policy")
        if isinstance(utility_with_review_summary.get("promotion_policy"), dict)
        else {},
        "rule_gap_observed": bool(
            applied_count == 0 and reusable_prefix_request_count == 0 and accepted_candidate_count > 0
        ),
        "review_queue_observed": bool(review_candidate_count > 0),
        "interpretation": (
            "Accepted semantic candidates exist while the current planner formed no reusable prefix. "
            "This is offline evidence of a rule coverage gap, not provider cache gain."
        )
        if applied_count == 0 and reusable_prefix_request_count == 0 and accepted_candidate_count > 0
        else "No accepted-candidate/no-prefix gap was observed in this offline run.",
    }


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


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
