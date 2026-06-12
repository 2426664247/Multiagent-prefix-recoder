from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .ir import BlockPlacement, CompileResult, PromptBlock
from .planner import PrefixPlan
from .validator import ValidationReport

FeedbackSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class PlacementFeedbackStats:
    placement_key: str
    attempts: int
    accepted: int
    rejected: int
    utility_failures: int
    cache_failures: int

    @property
    def acceptance_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.accepted / self.attempts

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["acceptance_rate"] = self.acceptance_rate
        return data


@dataclass(frozen=True)
class CandidateFeedbackRecord:
    run_id: str
    candidate_id: str
    scenario_id: str
    agent_ids: tuple[str, ...]
    accepted: bool
    applied: bool
    fallback_reason: str | None
    hard_constraint_passed: bool
    utility_status: str
    cache_hit_increased: bool | None
    estimated_cache_gain: float | None
    cached_tokens_delta: int | None
    latency_delta: float | None
    cost_delta: float | None
    validator_report: Mapping[str, Any] | None
    created_at: str


@dataclass(frozen=True)
class PlacementFeedbackRecord:
    run_id: str
    candidate_id: str
    placement_id: str
    block_id: str
    block_hash: str
    original_scope: str
    target_scope: str
    target_agent_group: tuple[str, ...]
    risk_tags: tuple[str, ...]
    dependency_notes: tuple[str, ...]
    cache_contribution: float
    hard_constraint_result: str
    utility_result: str
    cache_result: str
    placement_label: str


@dataclass(frozen=True)
class PlannerFeedbackRecord:
    schema_version: str
    prompt_safe_summary: bool
    session_id: str
    run_id: str
    candidate_id: str
    accepted: bool
    applied: bool
    fallback: bool
    validation_reason: str
    utility_preserved: bool | None
    utility_confidence: float | None
    utility_reason: str | None
    utility_status: str
    cache_hit_increased: bool | None
    estimated_gain_chars: int | None
    estimated_cache_gain: float | None
    moved_block_count: int
    moved_block_keys: tuple[str, ...]
    candidate_feedback: Mapping[str, Any]
    placement_feedback: tuple[Mapping[str, Any], ...]
    placement_outcomes: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JsonlFeedbackLogger:
    path: str | Path

    def __call__(self, record: Mapping[str, Any]) -> None:
        target = Path(self.path)
        if target.parent != Path("."):
            target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


class PlannerFeedbackLearner:
    """Prompt-safe placement-level feedback memory. It records labels but never trains a model."""

    def __init__(self, *, feedback_sink: FeedbackSink | None = None) -> None:
        self.records: list[PlannerFeedbackRecord] = []
        self.candidate_records: list[CandidateFeedbackRecord] = []
        self.placement_records: list[PlacementFeedbackRecord] = []
        self._sinks: list[FeedbackSink] = []
        if feedback_sink is not None:
            self._sinks.append(feedback_sink)
        self._attempts: Counter[str] = Counter()
        self._accepted: Counter[str] = Counter()
        self._weak_positive: Counter[str] = Counter()
        self._rejected: Counter[str] = Counter()
        self._utility_failures: Counter[str] = Counter()
        self._cache_failures: Counter[str] = Counter()

    def record(
        self,
        *,
        compile_result: CompileResult,
        plan: PrefixPlan,
        validation_report: ValidationReport,
    ) -> PlannerFeedbackRecord:
        utility_report = validation_report.utility_preservation_report
        utility_preserved = utility_report.is_utility_preserved if utility_report is not None else None
        utility_confidence = utility_report.confidence if utility_report is not None else None
        utility_reason = utility_report.reason if utility_report is not None else None
        utility_status = validation_report.utility_status
        cache_hit_increased = validation_report.cache_hit_increased
        cache_estimate_report = validation_report.cache_estimate_report
        accepted = bool(
            not validation_report.fallback
            and (not plan.moved_blocks or cache_hit_increased is True or validation_report.reason == "no_rewrite_needed")
        )
        candidate_id = (
            plan.prefix_tree_candidate.candidate_id
            if plan.prefix_tree_candidate is not None
            else f"{compile_result.session_id}:candidate:legacy"
        )
        run_id = f"{compile_result.session_id}:{len(self.records) + 1:06d}"
        agent_ids = tuple(plan.prefix_tree_candidate.agent_paths.keys()) if plan.prefix_tree_candidate else (compile_result.session_id,)
        candidate_record = CandidateFeedbackRecord(
            run_id=run_id,
            candidate_id=candidate_id,
            scenario_id=compile_result.session_id,
            agent_ids=agent_ids,
            accepted=accepted,
            applied=validation_report.applied,
            fallback_reason=validation_report.reason if validation_report.fallback else None,
            hard_constraint_passed=validation_report.hard_constraint_passed,
            utility_status=utility_status,
            cache_hit_increased=cache_hit_increased,
            estimated_cache_gain=(
                float(cache_estimate_report.estimated_cache_gain)
                if cache_estimate_report is not None
                else plan.estimated_cache_gain
            ),
            cached_tokens_delta=cache_estimate_report.cached_tokens_delta if cache_estimate_report is not None else None,
            latency_delta=None,
            cost_delta=None,
            validator_report={
                "reason": validation_report.reason,
                "hard_constraint_report": validation_report.hard_constraint_report,
                "utility_preservation": asdict(utility_report) if utility_report is not None else None,
                "cache_estimate_report": asdict(cache_estimate_report) if cache_estimate_report is not None else None,
            },
            created_at=_utc_now_iso(),
        )

        placement_outcomes: list[dict[str, Any]] = []
        placement_records: list[PlacementFeedbackRecord] = []
        blocks_by_id = {block.block_id: block for block in compile_result.blocks}
        placements = plan.placements or _legacy_placements_from_plan(plan, compile_result)
        for placement in placements:
            block = blocks_by_id.get(placement.block_id)
            placement_key = placement_key_for_placement(placement, block=block)
            label = _placement_label(
                validation_report=validation_report,
                placement=placement,
                accepted=accepted,
            )
            hard_result = "passed" if validation_report.hard_constraint_passed else "failed"
            cache_result = (
                "positive"
                if placement.cache_contribution > 0 and cache_hit_increased is True
                else "negative"
                if cache_hit_increased is False and placement.moved
                else "not_applicable"
            )
            utility_result = utility_status
            placement_record = PlacementFeedbackRecord(
                run_id=run_id,
                candidate_id=candidate_id,
                placement_id=placement.placement_id,
                block_id=placement.block_id,
                block_hash=placement.block_hash,
                original_scope=_scope_value(placement.original_scope),
                target_scope=_scope_value(placement.target_scope),
                target_agent_group=placement.target_agent_group,
                risk_tags=placement.risk_tags,
                dependency_notes=placement.dependency_notes,
                cache_contribution=placement.cache_contribution,
                hard_constraint_result=hard_result,
                utility_result=utility_result,
                cache_result=cache_result,
                placement_label=label,
            )
            placement_records.append(placement_record)
            outcome = {
                "placement_key": placement_key,
                "placement_id": placement.placement_id,
                "block_id": placement.block_id,
                "block_hash": placement.block_hash,
                "semantic_hint": block.semantic_type.value if block is not None else "unknown",
                "movability": block.movability.value if block is not None else "unknown",
                "original_scope": _scope_value(placement.original_scope),
                "target_scope": _scope_value(placement.target_scope),
                "risk_tags": placement.risk_tags,
                "accepted": accepted,
                "validation_reason": validation_report.reason,
                "placement_label": label,
            }
            placement_outcomes.append(outcome)
            self._attempts[placement_key] += 1
            if accepted:
                self._accepted[placement_key] += 1
            if label == "weak_positive":
                self._weak_positive[placement_key] += 1
            if label in {"hard_rejected", "utility_negative"}:
                self._rejected[placement_key] += 1
            if label == "utility_negative":
                self._utility_failures[placement_key] += 1
            if label == "cache_negative":
                self._cache_failures[placement_key] += 1

        record = PlannerFeedbackRecord(
            schema_version="prefix-planner-feedback-record-v2",
            prompt_safe_summary=True,
            session_id=compile_result.session_id,
            run_id=run_id,
            candidate_id=candidate_id,
            accepted=accepted,
            applied=validation_report.applied,
            fallback=validation_report.fallback,
            validation_reason=validation_report.reason,
            utility_preserved=utility_preserved,
            utility_confidence=utility_confidence,
            utility_reason=utility_reason,
            utility_status=utility_status,
            cache_hit_increased=cache_hit_increased,
            estimated_gain_chars=(
                validation_report.utility_estimate.estimated_gain_chars
                if validation_report.utility_estimate is not None
                else None
            ),
            estimated_cache_gain=plan.estimated_cache_gain,
            moved_block_count=len(plan.moved_blocks),
            moved_block_keys=tuple(outcome["placement_key"] for outcome in placement_outcomes if outcome["block_id"] in plan.moved_blocks),
            candidate_feedback=asdict(candidate_record),
            placement_feedback=tuple(asdict(item) for item in placement_records),
            placement_outcomes=tuple(placement_outcomes),
        )
        self.records.append(record)
        self.candidate_records.append(candidate_record)
        self.placement_records.extend(placement_records)
        for sink in self._sinks:
            sink(record.to_dict())
        return record

    def success_rate_for(self, block: PromptBlock) -> float | None:
        key = placement_key_for_block(block)
        attempts = self._attempts.get(key, 0)
        if attempts == 0:
            return None
        return self._accepted.get(key, 0) / attempts

    def success_rate_for_placement(self, **features: Any) -> float | None:
        key = placement_key_from_features(**features)
        attempts = self._attempts.get(key, 0)
        if attempts == 0:
            return None
        return self._accepted.get(key, 0) / attempts

    def weak_success_rate_for_placement(self, **features: Any) -> float | None:
        key = placement_key_from_features(**features)
        attempts = self._attempts.get(key, 0)
        if attempts == 0:
            return None
        return self._weak_positive.get(key, 0) / attempts

    def rejection_rate_for_placement(self, **features: Any) -> float | None:
        key = placement_key_from_features(**features)
        attempts = self._attempts.get(key, 0)
        if attempts == 0:
            return None
        return self._rejected.get(key, 0) / attempts

    def get_historical_score_for_placement(self, **features: Any) -> float | None:
        weak = self.weak_success_rate_for_placement(**features)
        success = self.success_rate_for_placement(**features)
        rejection = self.rejection_rate_for_placement(**features)
        values = [value for value in (weak, success) if value is not None]
        if not values and rejection is None:
            return None
        positive = sum(values) / len(values) if values else 0.0
        negative = rejection or 0.0
        return max(0.0, min(1.0, 0.5 + 0.5 * positive - 0.5 * negative))

    def get_feedback_summary_by_scope(self) -> dict[str, Any]:
        scope_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for record in self.placement_records:
            scope_counts[record.target_scope][record.placement_label] += 1
        return {
            "schema_version": "prefix-placement-feedback-summary-by-scope-v1",
            "prompt_safe_summary": True,
            "scope_label_counts": {scope: dict(counts) for scope, counts in sorted(scope_counts.items())},
        }

    def export_replay_dataset(self) -> tuple[dict[str, Any], ...]:
        return tuple(record.to_dict() for record in self.records)

    def summarize(self) -> dict[str, Any]:
        placement_stats = tuple(
            self._placement_stats(key).to_dict()
            for key in sorted(self._attempts)
        )
        reason_counts = Counter(record.validation_reason for record in self.records)
        label_counts = Counter(record.placement_label for record in self.placement_records)
        return {
            "schema_version": "prefix-planner-feedback-summary-v2",
            "prompt_safe_summary": True,
            "record_count": len(self.records),
            "candidate_record_count": len(self.candidate_records),
            "placement_record_count": len(self.placement_records),
            "accepted_count": sum(1 for record in self.records if record.accepted),
            "fallback_count": sum(1 for record in self.records if record.fallback),
            "validation_reason_counts": dict(sorted(reason_counts.items())),
            "placement_label_counts": dict(sorted(label_counts.items())),
            "placement_stats": placement_stats,
            "summary_by_scope": self.get_feedback_summary_by_scope(),
            "limits": (
                "This feedback is prompt-safe metadata for planner policy learning. "
                "It does not train a model or prove provider cache, latency, cost, or task success."
            ),
        }

    def _placement_stats(self, key: str) -> PlacementFeedbackStats:
        attempts = self._attempts.get(key, 0)
        accepted = self._accepted.get(key, 0)
        utility_failures = self._utility_failures.get(key, 0)
        cache_failures = self._cache_failures.get(key, 0)
        return PlacementFeedbackStats(
            placement_key=key,
            attempts=attempts,
            accepted=accepted,
            rejected=max(0, attempts - accepted),
            utility_failures=utility_failures,
            cache_failures=cache_failures,
        )


def placement_key_for_block(block: PromptBlock) -> str:
    return placement_key_from_features(
        semantic_hint=block.semantic_type.value,
        movability=block.movability.value,
        original_scope=block.share_scope.value,
        target_scope=block.share_scope.value,
        risk_tags=block.risk_tags,
    )


def placement_key_for_placement(placement: BlockPlacement, *, block: PromptBlock | None = None) -> str:
    return placement_key_from_features(
        semantic_hint=block.semantic_type.value if block is not None else "unknown",
        movability=block.movability.value if block is not None else "unknown",
        original_scope=_scope_value(placement.original_scope),
        target_scope=_scope_value(placement.target_scope),
        risk_tags=placement.risk_tags,
    )


def placement_key_from_features(**features: Any) -> str:
    risk_tags = features.get("risk_tags") or ()
    if isinstance(risk_tags, str):
        risk_key = risk_tags
    else:
        risk_key = ",".join(sorted(str(tag) for tag in risk_tags)) or "none"
    return (
        f"semantic={features.get('semantic_hint') or features.get('semantic_type') or 'unknown'}|"
        f"movability={features.get('movability') or 'unknown'}|"
        f"original_scope={_scope_value(features.get('original_scope'))}|"
        f"target_scope={_scope_value(features.get('target_scope'))}|"
        f"risk={risk_key}"
    )


def summarize_feedback_records(records: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]]) -> dict[str, Any]:
    reason_counts = Counter(str(record.get("validation_reason") or "unknown") for record in records)
    placement_attempts: dict[str, int] = defaultdict(int)
    placement_accepted: dict[str, int] = defaultdict(int)
    placement_labels: Counter[str] = Counter()
    for record in records:
        outcomes = record.get("placement_outcomes")
        if not isinstance(outcomes, (list, tuple)):
            continue
        for outcome in outcomes:
            if not isinstance(outcome, Mapping):
                continue
            key = str(outcome.get("placement_key") or "unknown")
            placement_attempts[key] += 1
            if outcome.get("accepted") is True:
                placement_accepted[key] += 1
            placement_labels[str(outcome.get("placement_label") or "unknown")] += 1
    return {
        "schema_version": "prefix-planner-feedback-summary-v2",
        "prompt_safe_summary": True,
        "record_count": len(records),
        "accepted_count": sum(1 for record in records if record.get("accepted") is True),
        "fallback_count": sum(1 for record in records if record.get("fallback") is True),
        "validation_reason_counts": dict(sorted(reason_counts.items())),
        "placement_label_counts": dict(sorted(placement_labels.items())),
        "placement_stats": tuple(
            PlacementFeedbackStats(
                placement_key=key,
                attempts=placement_attempts[key],
                accepted=placement_accepted.get(key, 0),
                rejected=max(0, placement_attempts[key] - placement_accepted.get(key, 0)),
                utility_failures=0,
                cache_failures=0,
            ).to_dict()
            for key in sorted(placement_attempts)
        ),
    }


def load_jsonl_feedback(path: str | Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if isinstance(value, dict):
                records.append(value)
    return tuple(records)


def _placement_label(
    *,
    validation_report: ValidationReport,
    placement: BlockPlacement,
    accepted: bool,
) -> str:
    if not validation_report.hard_constraint_passed:
        return "hard_rejected"
    if validation_report.utility_status == "failed":
        return "utility_negative"
    if validation_report.cache_hit_increased is False and placement.moved:
        return "cache_negative"
    if validation_report.utility_status == "passed" and accepted:
        return "utility_positive"
    if accepted:
        return "weak_positive"
    return "hard_rejected" if validation_report.fallback else "cache_negative"


def _legacy_placements_from_plan(plan: PrefixPlan, compile_result: CompileResult) -> tuple[BlockPlacement, ...]:
    placements: list[BlockPlacement] = []
    for block in compile_result.blocks:
        placements.append(
            BlockPlacement(
                placement_id=f"{compile_result.session_id}:legacy:{block.block_id}",
                block_id=block.block_id,
                block_hash=block.content_hash,
                original_agent_id=block.agent_or_source or compile_result.session_id,
                original_position=block.original_position,
                original_scope=block.share_scope,
                target_scope=block.share_scope.value,
                target_node_id="legacy",
                target_agent_group=(block.agent_or_source or compile_result.session_id,),
                moved=block.block_id in plan.moved_blocks,
                risk_tags=block.risk_tags,
                dependency_notes=block.dependency_refs,
            )
        )
    return tuple(placements)


def _scope_value(scope: Any) -> str:
    value = getattr(scope, "value", scope)
    if value == "agent":
        return "agent_local"
    if value is None:
        return "unknown"
    return str(value)


def _utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat()
