from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ir import CompileResult, PrefixTreeCandidate, stable_hash
from .planner import PrefixPlan
from .telemetry import dataclass_to_dict, serialize_prefix_tree_candidate
from .validator import CacheUtilityValidator, ValidationReport


@dataclass(frozen=True)
class ReplayRunStore:
    root_dir: str | Path

    def save_replay_run(
        self,
        *,
        run_id: str,
        compile_result: CompileResult,
        plan: PrefixPlan,
        validation_report: ValidationReport,
        final_decision: str,
        fallback_reason: str | None = None,
        feedback_records: Sequence[Mapping[str, Any]] = (),
        include_text: bool = False,
    ) -> Path:
        target = Path(self.root_dir) / f"{_safe_id(run_id)}.json"
        if target.parent != Path("."):
            target.parent.mkdir(parents=True, exist_ok=True)
        candidates = (plan.prefix_tree_candidate,) if plan.prefix_tree_candidate is not None else ()
        value = build_replay_run_record(
            run_id=run_id,
            compile_result=compile_result,
            plan=plan,
            validation_report=validation_report,
            candidates=candidates,
            final_decision=final_decision,
            fallback_reason=fallback_reason,
            feedback_records=feedback_records,
            include_text=include_text,
        )
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return target

    def load_replay_run(self, run_id: str) -> dict[str, Any]:
        return load_replay_run(Path(self.root_dir) / f"{_safe_id(run_id)}.json")


def build_replay_run_record(
    *,
    run_id: str,
    compile_result: CompileResult,
    plan: PrefixPlan,
    validation_report: ValidationReport,
    candidates: Sequence[PrefixTreeCandidate],
    final_decision: str,
    fallback_reason: str | None = None,
    feedback_records: Sequence[Mapping[str, Any]] = (),
    include_text: bool = False,
) -> dict[str, Any]:
    original_prompts = _message_summaries(compile_result.messages, include_text=include_text)
    original_prompt_hashes = tuple(
        stable_hash(
            {
                "type": getattr(message, "type", type(message).__name__),
                "source": getattr(message, "source", None),
                "content": getattr(message, "content", None),
            }
        )
        for message in compile_result.messages
    )
    return {
        "schema_version": "prefix-replay-run-v1",
        "run_id": run_id,
        "session_id": compile_result.session_id,
        "prompt_text_included": include_text,
        "original_prompts": original_prompts,
        "original_prompt_hashes": original_prompt_hashes,
        "compiler_blocks": tuple(_block_metadata(block, include_text=include_text) for block in compile_result.blocks),
        "block_metadata": tuple(_block_metadata(block, include_text=False) for block in compile_result.blocks),
        "prefix_tree_candidates": tuple(
            serialize_prefix_tree_candidate(candidate, include_text=include_text) for candidate in candidates
        ),
        "materialized_prompts": {
            candidate.candidate_id: (
                dict(candidate.materialized_prompts or {}) if include_text else _materialized_prompt_hashes(candidate)
            )
            for candidate in candidates
        },
        "block_placements": tuple(asdict(placement) for placement in plan.placements),
        "planner_score_breakdown": plan.planner_score_breakdown,
        "hard_validator_report": validation_report.hard_constraint_report,
        "utility_validator_report": dataclass_to_dict(validation_report.utility_preservation_report),
        "cache_gain_report": plan.cache_gain_report,
        "cache_estimate_report": dataclass_to_dict(validation_report.cache_estimate_report),
        "final_decision": final_decision,
        "fallback_reason": fallback_reason,
        "feedback_records": tuple(feedback_records),
    }


def load_replay_run(run_id_or_path: str | Path) -> dict[str, Any]:
    path = Path(run_id_or_path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def replay_candidate(candidate_id: str, *, replay_record: Mapping[str, Any] | None = None, replay_path: str | Path | None = None) -> dict[str, Any]:
    record = replay_record if replay_record is not None else load_replay_run(_require_path(replay_path))
    candidates = record.get("prefix_tree_candidates") or ()
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate.get("candidate_id") == candidate_id:
            return dict(candidate)
    raise KeyError(f"candidate not found in replay run: {candidate_id}")


def revalidate_candidate(
    candidate_id: str,
    validator: CacheUtilityValidator,
    *,
    replay_record: Mapping[str, Any] | None = None,
    replay_path: str | Path | None = None,
) -> dict[str, Any]:
    candidate = replay_candidate(candidate_id, replay_record=replay_record, replay_path=replay_path)
    return {
        "schema_version": "prefix-replay-revalidation-placeholder-v1",
        "candidate_id": candidate_id,
        "validator": type(validator).__name__,
        "utility_status": "unverified",
        "reason": (
            "Replay metadata loaded. Full message-object revalidation requires reconstructing AutoGen "
            "LLMMessage objects from a text-included replay artifact."
        ),
        "candidate": candidate,
    }


def export_utility_labeling_samples(
    replay_records: Sequence[Mapping[str, Any]],
    *,
    include_text: bool = False,
) -> tuple[dict[str, Any], ...]:
    samples: list[dict[str, Any]] = []
    for record in replay_records:
        for candidate in record.get("prefix_tree_candidates") or ():
            if not isinstance(candidate, Mapping):
                continue
            samples.append(
                {
                    "schema_version": "prefix-utility-labeling-sample-v1",
                    "run_id": record.get("run_id"),
                    "candidate_id": candidate.get("candidate_id"),
                    "prefix_tree_candidate": candidate,
                    "materialized_prompts": (
                        record.get("materialized_prompts", {}).get(candidate.get("candidate_id"))
                        if include_text
                        else None
                    ),
                    "hard_constraint_passed": (record.get("hard_validator_report") or {}).get("passed"),
                    "utility_status": (record.get("utility_validator_report") or {}).get("utility_status", "unverified"),
                    "expected_is_utility_preserved": None,
                    "label_status": "unlabeled",
                }
            )
    return tuple(samples)


def export_planner_training_samples(replay_records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    samples: list[dict[str, Any]] = []
    for record in replay_records:
        for feedback in record.get("feedback_records") or ():
            if not isinstance(feedback, Mapping):
                continue
            samples.append(
                {
                    "schema_version": "prefix-planner-training-sample-v1",
                    "run_id": record.get("run_id"),
                    "candidate_id": feedback.get("candidate_id"),
                    "candidate_feedback": feedback.get("candidate_feedback"),
                    "placement_feedback": feedback.get("placement_feedback"),
                    "planner_score_breakdown": record.get("planner_score_breakdown"),
                }
            )
    return tuple(samples)


def _message_summaries(messages: Sequence[Any], *, include_text: bool) -> tuple[dict[str, Any], ...]:
    summaries: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        content = getattr(message, "content", None)
        row = {
            "index": index,
            "type": getattr(message, "type", type(message).__name__),
            "source": getattr(message, "source", None),
            "content_hash": stable_hash(content),
        }
        if include_text:
            row["content"] = content
        summaries.append(row)
    return tuple(summaries)


def _block_metadata(block: Any, *, include_text: bool) -> dict[str, Any]:
    row = {
        "block_id": block.block_id,
        "block_hash": block.content_hash,
        "original_agent_id": block.agent_or_source,
        "original_message_id": block.source_message_id,
        "original_position": dataclass_to_dict(block.original_position),
        "source_type": block.source_type,
        "token_len": block.token_len,
        "candidate_shared_agents": block.candidate_shared_agents,
        "risk_tags": block.risk_tags,
        "dependency_before": block.dependency_before,
        "dependency_after": block.dependency_after,
        "movable_hint": block.movable_hint,
        "summary": block.summary,
        "contains_private_info": block.contains_private_info,
        "contains_role_identity": block.contains_role_identity,
        "contains_tool_permission": block.contains_tool_permission,
        "contains_latest_user_instruction": block.contains_latest_user_instruction,
        "contains_tool_result": block.contains_tool_result,
        "contains_credential": block.contains_credential,
    }
    if include_text:
        row["raw_text"] = block.rendered_text
    else:
        row["secure_text_ref"] = block.secure_text_ref or f"text_hash:{block.content_hash[:16]}"
    return row


def _materialized_prompt_hashes(candidate: PrefixTreeCandidate) -> dict[str, str]:
    return {
        agent_id: stable_hash(prompt)
        for agent_id, prompt in (candidate.materialized_prompts or {}).items()
    }


def _require_path(path: str | Path | None) -> str | Path:
    if path is None:
        raise ValueError("replay_path is required when replay_record is not provided")
    return path


def _safe_id(value: Any) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))[:180] or "run"
