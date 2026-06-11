from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ir import Movability, PromptBlock, SemanticType


@dataclass(frozen=True)
class ShadowTrialRule:
    trial_rule_id: str
    feature_set: str
    features: Mapping[str, Any]
    shadow_action: str
    support: int
    purity: float | None


@dataclass(frozen=True)
class ShadowTrialPlan:
    schema_version: str
    trial_mode: str
    rules: tuple[ShadowTrialRule, ...]
    source_path: str | None = None


def load_shadow_trial_plan(path: str | Path) -> ShadowTrialPlan:
    source_path = Path(path)
    value = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("shadow trial plan must be a JSON object")
    plan = _extract_shadow_trial_plan(value)
    rules = []
    for index, row in enumerate(plan.get("trial_rules") if isinstance(plan.get("trial_rules"), list) else []):
        if not isinstance(row, Mapping):
            continue
        features = row.get("features") if isinstance(row.get("features"), Mapping) else {}
        rule_id = str(row.get("trial_rule_id") or f"shadow_rule:{index}")
        shadow_action = str(row.get("shadow_action") or row.get("validator_action") or "")
        if not shadow_action:
            continue
        rules.append(
            ShadowTrialRule(
                trial_rule_id=rule_id,
                feature_set=str(row.get("feature_set") or "+".join(str(key) for key in features)),
                features={str(key): value for key, value in features.items()},
                shadow_action=shadow_action,
                support=_int(row.get("support")),
                purity=_float_or_none(row.get("purity")),
            )
        )
    return ShadowTrialPlan(
        schema_version=str(plan.get("schema_version") or "prefix-static-rule-shadow-trial-plan-v1"),
        trial_mode=str(plan.get("trial_mode") or "shadow_only"),
        rules=tuple(rules),
        source_path=str(source_path),
    )


def evaluate_shadow_trial(
    plan: ShadowTrialPlan | None,
    blocks: Sequence[PromptBlock],
) -> dict[str, Any] | None:
    if plan is None:
        return None
    matched_rules: list[dict[str, Any]] = []
    unsupported_rules: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    matched_block_ids: set[str] = set()
    for rule in plan.rules:
        unsupported = sorted(_unsupported_feature_names(rule.features))
        if unsupported:
            unsupported_rules.append(
                {
                    "trial_rule_id": rule.trial_rule_id,
                    "unsupported_feature_names": unsupported,
                    "shadow_action": rule.shadow_action,
                }
            )
            continue
        matched_blocks = [block for block in blocks if _rule_matches_block(rule, block)]
        if not matched_blocks:
            continue
        matched_block_ids.update(block.block_id for block in matched_blocks)
        action_counts[rule.shadow_action] += 1
        matched_rules.append(
            {
                "trial_rule_id": rule.trial_rule_id,
                "feature_set": rule.feature_set,
                "features": dict(rule.features),
                "shadow_action": rule.shadow_action,
                "support": rule.support,
                "purity": rule.purity,
                "matched_block_count": len(matched_blocks),
                "matched_block_ids": tuple(block.block_id for block in matched_blocks[:8]),
                "matched_semantic_type_counts": dict(
                    sorted(Counter(block.semantic_type.value for block in matched_blocks).items())
                ),
                "matched_risk_tag_counts": _risk_tag_counts(matched_blocks),
            }
        )
    return {
        "schema_version": "prefix-shadow-trial-telemetry-v1",
        "prompt_safe_summary": True,
        "enabled": True,
        "trial_mode": plan.trial_mode,
        "plan_schema_version": plan.schema_version,
        "plan_source_path": plan.source_path,
        "trial_rule_count": len(plan.rules),
        "matched_rule_count": len(matched_rules),
        "matched_block_count": len(matched_block_ids),
        "trial_action_counts": dict(sorted(action_counts.items())),
        "unsupported_rule_count": len(unsupported_rules),
        "unsupported_rules": unsupported_rules[:12],
        "matched_rules": matched_rules[:20],
        "validator_behavior_change_allowed": False,
        "safe_for_automatic_validator_promotion": False,
        "automation_policy": "shadow_only_no_behavior_change",
        "limits": (
            "Shadow trial telemetry is metadata-only. It records rule matches and never changes request rewriting, "
            "Validator decisions, provider calls, or benchmark scoring."
        ),
    }


def _extract_shadow_trial_plan(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if value.get("schema_version") == "prefix-static-rule-shadow-trial-plan-v1":
        return value
    candidates = value.get("validator_rule_candidates")
    if isinstance(candidates, Mapping) and isinstance(candidates.get("shadow_trial_plan"), Mapping):
        return candidates["shadow_trial_plan"]
    if isinstance(value.get("shadow_trial_plan"), Mapping):
        return value["shadow_trial_plan"]
    raise ValueError("JSON does not contain a shadow_trial_plan")


def _rule_matches_block(rule: ShadowTrialRule, block: PromptBlock) -> bool:
    return all(_feature_matches_block(name, value, block) for name, value in rule.features.items())


def _feature_matches_block(name: str, value: Any, block: PromptBlock) -> bool:
    normalized = str(value or "missing")
    if name == "semantic_hint":
        return _block_semantic_hint(block) == normalized
    if name == "risk_tag_set":
        return _block_risk_tag_set(block) == normalized
    return False


def _unsupported_feature_names(features: Mapping[str, Any]) -> set[str]:
    return {str(name) for name in features if str(name) not in {"semantic_hint", "risk_tag_set"}}


def _block_semantic_hint(block: PromptBlock) -> str:
    risk_tags = set(str(tag) for tag in block.risk_tags)
    if "natural_language_verification_policy" in risk_tags:
        return "verification_policy"
    if "natural_language_procedure_policy" in risk_tags:
        return "procedure_policy"
    if "natural_language_tool_or_code_policy" in risk_tags:
        return "tool_or_code_policy"
    if "natural_language_team_policy" in risk_tags or "team_policy" in risk_tags:
        return "team_policy"
    if "tool_schema" in risk_tags or block.semantic_type == SemanticType.SHARED_TOOL_DESCRIPTION:
        return "tool_or_code_policy"
    if block.semantic_type == SemanticType.TEAM_POLICY:
        return "team_policy"
    if block.semantic_type == SemanticType.OUTPUT_FORMAT:
        return "long_stable_instruction"
    return block.semantic_type.value


def _block_risk_tag_set(block: PromptBlock) -> str:
    risk_tags = set(str(tag) for tag in block.risk_tags)
    mapped: set[str] = set()
    if block.semantic_type == SemanticType.ROLE_IDENTITY or "agent_identity" in risk_tags:
        mapped.add("agent_identity_boundary")
    if "conditional_instruction" in risk_tags or block.movability == Movability.CONDITIONAL_PREFIX:
        mapped.add("conditional_instruction")
    if block.semantic_type in {SemanticType.CURRENT_TURN_INSTRUCTION, SemanticType.CURRENT_USER_INSTRUCTION}:
        mapped.add("possible_turn_specific_reference")
    if "latest_user_instruction" in risk_tags or "current_turn_instruction" in risk_tags:
        mapped.add("possible_turn_specific_reference")
    return "|".join(sorted(mapped)) if mapped else "none"


def _risk_tag_counts(blocks: Sequence[PromptBlock]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for block in blocks:
        counter.update(str(tag) for tag in block.risk_tags)
    return dict(sorted(counter.items()))


def _int(value: Any) -> int:
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


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None
