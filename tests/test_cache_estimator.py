from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from autogen_core.models import LLMMessage, SystemMessage

from autogen_prefix_tree import (
    CacheEstimateReport,
    CacheHitProxyEstimator,
    CacheUtilityValidator,
    HierarchicalPrefixPlanner,
    LocalPromptCompiler,
    PrefixReorderClient,
    PrefixTreeEstimator,
    build_replay_run_record,
    rewrite_messages,
    UtilityPreservationReport,
)
from autogen_prefix_tree.cache_estimator_smoke import run_smoke
from autogen_prefix_tree.static_client import StaticResponseClient


class FixedCacheEstimator:
    name = "fixed_cache_estimator"

    def __init__(self, report: CacheEstimateReport) -> None:
        self.report = report
        self.calls = 0

    def estimate(
        self,
        original_requests: Sequence[Any],
        rewritten_requests: Sequence[Any],
        candidate: Any | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> CacheEstimateReport:
        del original_requests, rewritten_requests, candidate, context
        self.calls += 1
        return self.report


class FixedUtilityModelValidator:
    threshold = 0.6
    model_name = "fixed-utility-model"

    def __init__(self, *, prediction: bool | None, confidence: float) -> None:
        self.prediction = prediction
        self.confidence = confidence
        self.calls = 0

    def evaluate(self, **_: Any) -> UtilityPreservationReport:
        self.calls += 1
        return UtilityPreservationReport(
            is_utility_preserved=self.prediction,
            confidence=self.confidence,
            reason="fixed_structured_shadow_prediction",
            checks=("structured_features_only",),
            model_name=self.model_name,
            prompt_safe=True,
            utility_status="passed" if self.prediction is True else "failed" if self.prediction is False else "unverified",
        )


def test_prefix_tree_estimator_outputs_unified_report() -> None:
    warm, plan = _warm_plan()

    report = PrefixTreeEstimator().estimate(
        warm.messages,
        rewrite_messages(warm, plan),
        candidate=plan.prefix_tree_candidate,
        context={"compile_result": warm, "plan": plan},
    )

    assert report.estimator_name == "prefix_tree"
    assert report.estimator_available is True
    assert report.used_fallback is False
    assert report.estimated_cache_gain > 0
    assert report.rewritten_cacheable_tokens_or_chars >= report.original_cacheable_tokens_or_chars
    assert report.node_contributions
    assert report.placement_contributions


def test_cache_hit_proxy_estimator_calls_local_proxy_function() -> None:
    warm, plan = _warm_plan(session_id="proxy-call")
    estimator = CacheHitProxyEstimator(block_size=1)
    estimator.observe(
        warm.messages,
        warm.messages,
        accepted=False,
        context={"compile_result": warm, "plan": plan, "session_id": "proxy-call"},
    )

    report = estimator.estimate(
        warm.messages,
        rewrite_messages(warm, plan),
        candidate=plan.prefix_tree_candidate,
        context={"compile_result": warm, "plan": plan, "session_id": "proxy-call"},
    )

    assert estimator.calls == 1
    assert report.estimator_name == "cache_hit_proxy"
    assert report.estimator_available is True
    assert report.used_fallback is False
    assert report.cached_tokens_delta is not None
    assert "offline_hash_token_units" in report.warnings


def test_cache_hit_proxy_estimator_falls_back_when_unavailable(tmp_path: Path) -> None:
    warm, plan = _warm_plan(session_id="proxy-missing")
    estimator = CacheHitProxyEstimator(
        estimator_module_path=tmp_path / "missing_cache_estimator.py",
        fallback_estimator=PrefixTreeEstimator(),
    )

    report = estimator.estimate(
        warm.messages,
        rewrite_messages(warm, plan),
        candidate=plan.prefix_tree_candidate,
        context={"compile_result": warm, "plan": plan},
    )

    assert report.estimator_name == "prefix_tree"
    assert report.used_fallback is True
    assert report.estimated_cache_gain > 0
    assert "cache_hit_proxy_adapter_failed" in report.warnings


def test_validator_cache_gate_uses_unified_estimator_report() -> None:
    warm, plan = _warm_plan(session_id="validator-fixed")
    estimator = FixedCacheEstimator(
        CacheEstimateReport(
            estimator_name="fixed_cache_estimator",
            estimator_available=True,
            cached_tokens_delta=0,
            estimated_cache_gain=0,
            reason="fixed_no_gain",
        )
    )
    validator = CacheUtilityValidator(cache_estimator=estimator)

    report = validator.validate(
        original_messages=warm.messages,
        rewritten_messages=rewrite_messages(warm, plan),
        compile_result=warm,
        plan=plan,
    )

    assert estimator.calls == 1
    assert report.fallback is True
    assert report.reason == "insufficient_cache_utility"
    assert report.cache_hit_increased is False
    assert report.cache_estimate_report is not None
    assert report.cache_estimate_report.estimator_name == "fixed_cache_estimator"


def test_hard_gate_failure_does_not_call_configured_cache_estimator() -> None:
    warm, plan = _warm_plan(session_id="hard-gate-first")
    estimator = FixedCacheEstimator(CacheEstimateReport(estimator_name="fixed", estimator_available=True))
    bad_plan = type(plan)(
        original_order=plan.original_order,
        new_order=plan.new_order[:-1],
        moved_blocks=plan.moved_blocks,
        kept_blocks=plan.kept_blocks,
        move_reason=plan.move_reason,
        cacheable_prefix_blocks=plan.cacheable_prefix_blocks,
        prefix_tree=plan.prefix_tree,
        prefix_tree_candidate=plan.prefix_tree_candidate,
        placements=plan.placements,
        estimated_cache_gain=plan.estimated_cache_gain,
        planner_score=plan.planner_score,
        planner_score_breakdown=plan.planner_score_breakdown,
        cache_gain_report=plan.cache_gain_report,
        risk_notes=plan.risk_notes,
    )

    report = CacheUtilityValidator(cache_estimator=estimator).validate(
        original_messages=warm.messages,
        rewritten_messages=rewrite_messages(warm, plan),
        compile_result=warm,
        plan=bad_plan,
    )

    assert report.fallback is True
    assert report.reason == "block_id_multiset_changed"
    assert estimator.calls == 0


def test_client_telemetry_records_cache_estimate_without_prompt_text() -> None:
    records: list[dict[str, Any]] = []
    inner = StaticResponseClient(responses=("OK",))
    client = PrefixReorderClient(
        inner,
        session_id="telemetry-cache-estimator",
        cache_estimator="cache_hit_proxy",
        telemetry_sink=records.append,
    )

    import asyncio

    asyncio.run(client.create(_messages("planner")))
    asyncio.run(client.create(_messages("engineer")))

    cache_report = records[-1]["cache_estimate_report"]
    assert cache_report["estimator_name"] == "cache_hit_proxy"
    assert cache_report["estimator_available"] is True
    assert cache_report["used_fallback"] is False
    serialized = json.dumps(records[-1], ensure_ascii=False)
    assert "AGENT_NAME: engineer" not in serialized
    assert "Shared estimator context." not in serialized


def test_replay_records_cache_estimate_without_prompt_text() -> None:
    warm, plan = _warm_plan(session_id="replay-cache-estimator")
    validator = CacheUtilityValidator(cache_estimator="cache_hit_proxy")
    report = validator.validate(
        original_messages=warm.messages,
        rewritten_messages=rewrite_messages(warm, plan),
        compile_result=warm,
        plan=plan,
    )

    replay = build_replay_run_record(
        run_id="cache-estimator-replay",
        compile_result=warm,
        plan=plan,
        validation_report=report,
        candidates=(plan.prefix_tree_candidate,),
        final_decision="applied",
        include_text=False,
    )

    assert replay["cache_estimate_report"]["estimator_name"] in {"cache_hit_proxy", "prefix_tree"}
    serialized = json.dumps(replay, ensure_ascii=False)
    assert "AGENT_NAME: engineer" not in serialized
    assert "Shared estimator context." not in serialized


def test_utility_model_shadow_records_fields_without_changing_default_decision() -> None:
    warm, plan = _warm_plan(session_id="utility-model-shadow")
    rewritten_messages = rewrite_messages(warm, plan)
    default_report = CacheUtilityValidator().validate(
        original_messages=warm.messages,
        rewritten_messages=rewritten_messages,
        compile_result=warm,
        plan=plan,
    )
    utility_model = FixedUtilityModelValidator(prediction=False, confidence=0.91)
    shadow_report = CacheUtilityValidator(
        utility_model_validator=utility_model,
        utility_model_mode="shadow",
        utility_model_min_confidence=0.6,
    ).validate(
        original_messages=warm.messages,
        rewritten_messages=rewritten_messages,
        compile_result=warm,
        plan=plan,
    )

    assert utility_model.calls == 1
    assert (shadow_report.applied, shadow_report.fallback, shadow_report.reason) == (
        default_report.applied,
        default_report.fallback,
        default_report.reason,
    )
    assert shadow_report.utility_model_gate_report is not None
    gate = shadow_report.utility_model_gate_report
    assert gate.utility_model_prediction is False
    assert gate.utility_model_confidence == 0.91
    assert gate.utility_model_threshold == 0.6
    assert gate.utility_model_shadow_decision == "would_reject"
    assert gate.hard_gate_result == "passed"
    assert gate.cache_gate_result == "passed"

    replay = build_replay_run_record(
        run_id="utility-model-shadow",
        compile_result=warm,
        plan=plan,
        validation_report=shadow_report,
        candidates=(plan.prefix_tree_candidate,) if plan.prefix_tree_candidate is not None else (),
        final_decision="applied" if shadow_report.applied else "fallback",
        include_text=False,
    )
    for key in (
        "utility_model_prediction",
        "utility_model_confidence",
        "utility_model_threshold",
        "utility_model_shadow_decision",
        "hard_gate_result",
        "cache_gate_result",
    ):
        assert key in replay
    assert replay["utility_model_shadow_decision"] == "would_reject"
    serialized = json.dumps(replay, ensure_ascii=False)
    assert "AGENT_NAME: engineer" not in serialized
    assert "Shared estimator context." not in serialized


def test_utility_model_shadow_telemetry_records_fields_without_prompt_text() -> None:
    records: list[dict[str, Any]] = []
    utility_model = FixedUtilityModelValidator(prediction=True, confidence=0.88)
    client = PrefixReorderClient(
        StaticResponseClient(responses=("OK", "OK")),
        session_id="utility-model-telemetry",
        validator=CacheUtilityValidator(
            utility_model_validator=utility_model,
            utility_model_mode="shadow",
            utility_model_min_confidence=0.6,
        ),
        telemetry_sink=records.append,
    )

    import asyncio

    asyncio.run(client.create(_messages("planner")))
    asyncio.run(client.create(_messages("engineer")))

    telemetry = records[-1]
    assert telemetry["utility_model_prediction"] is True
    assert telemetry["utility_model_confidence"] == 0.88
    assert telemetry["utility_model_threshold"] == 0.6
    assert telemetry["utility_model_shadow_decision"] == "would_accept"
    assert telemetry["hard_gate_result"] == "passed"
    assert telemetry["cache_gate_result"] == "passed"
    assert telemetry["utility_model_gate"]["mode"] == "shadow"
    serialized = json.dumps(telemetry, ensure_ascii=False)
    assert "AGENT_NAME: engineer" not in serialized
    assert "Shared estimator context." not in serialized


def test_low_confidence_utility_model_observes_and_gate_does_not_block() -> None:
    warm, plan = _warm_plan(session_id="utility-model-observe")
    rewritten_messages = rewrite_messages(warm, plan)
    default_report = CacheUtilityValidator().validate(
        original_messages=warm.messages,
        rewritten_messages=rewritten_messages,
        compile_result=warm,
        plan=plan,
    )
    utility_model = FixedUtilityModelValidator(prediction=False, confidence=0.51)

    gate_report = CacheUtilityValidator(
        utility_model_validator=utility_model,
        utility_model_mode="gate",
        utility_model_min_confidence=0.8,
    ).validate(
        original_messages=warm.messages,
        rewritten_messages=rewritten_messages,
        compile_result=warm,
        plan=plan,
    )

    assert utility_model.calls == 1
    assert (gate_report.applied, gate_report.fallback, gate_report.reason) == (
        default_report.applied,
        default_report.fallback,
        default_report.reason,
    )
    assert gate_report.utility_model_gate_report is not None
    assert gate_report.utility_model_gate_report.utility_model_shadow_decision == "observe"
    assert gate_report.utility_model_gate_report.cache_gate_result == "passed"


def test_cache_estimator_smoke_cli_summary_prefix_tree(tmp_path: Path) -> None:
    summary = run_smoke(estimator="prefix-tree", output_dir=tmp_path / "prefix")

    assert summary["prompt_safe_summary"] is True
    assert summary["network_access_required"] is False
    assert summary["estimator_name"] == "prefix_tree"
    assert summary["validation_reason"] == "validated"
    assert (tmp_path / "prefix" / "summary.json").exists()


def test_cache_estimator_smoke_cli_summary_proxy(tmp_path: Path) -> None:
    summary = run_smoke(estimator="cache-hit-proxy", output_dir=tmp_path / "proxy")

    assert summary["prompt_safe_summary"] is True
    assert summary["network_access_required"] is False
    assert summary["estimator_name"] in {"cache_hit_proxy", "prefix_tree"}
    if summary["estimator_name"] == "prefix_tree":
        assert summary["used_fallback"] is True
    assert (tmp_path / "proxy" / "summary.json").exists()


def _warm_plan(session_id: str = "cache-estimator-test"):
    compiler = LocalPromptCompiler()
    planner = HierarchicalPrefixPlanner()
    cold = compiler.compile(_messages("planner"), session_id=session_id)
    planner.plan(cold, session_id=session_id)
    warm = compiler.compile(_messages("engineer"), session_id=session_id)
    plan = planner.plan(warm, session_id=session_id)
    return warm, plan


def _messages(agent: str) -> tuple[LLMMessage, ...]:
    return (
        SystemMessage(
            content="\n\n".join(
                [
                    "ROLE_SPECIFIC_INSTRUCTION_START\n"
                    f"AGENT_NAME: {agent}\n"
                    f"You are {agent}.\n"
                    "ROLE_SPECIFIC_INSTRUCTION_END",
                    "USER_TASK_START\nBuild the estimator integration.\nUSER_TASK_END",
                    "SHARED_GROUPCHAT_CONTEXT_START\nShared estimator context.\nSHARED_GROUPCHAT_CONTEXT_END",
                    "TOOL_SCHEMA_START\nshared_tool(x: string) -> string\nTOOL_SCHEMA_END",
                    "OUTPUT_FORMAT_START\nReturn a concise status.\nOUTPUT_FORMAT_END",
                    "CURRENT_TURN_INSTRUCTION_START\nAnswer this turn only.\nCURRENT_TURN_INSTRUCTION_END",
                ]
            )
        ),
    )
