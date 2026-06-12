from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from autogen_core.models import LLMMessage

from .ir import CompileResult, stable_hash
from .planner import PrefixPlan


@dataclass(frozen=True)
class CacheEstimateReport:
    schema_version: str = "cache-estimate-report-v1"
    estimator_name: str = "unknown"
    estimator_available: bool = False
    used_fallback: bool = False
    original_cacheable_tokens_or_chars: int = 0
    rewritten_cacheable_tokens_or_chars: int = 0
    cached_tokens_delta: int | None = None
    cache_hit_ratio_before: float | None = None
    cache_hit_ratio_after: float | None = None
    estimated_cache_gain: float = 0.0
    global_prefix_contribution: float = 0.0
    subgroup_prefix_contribution: float = 0.0
    lcp_contribution: float = 0.0
    node_contributions: Mapping[str, Any] | None = None
    placement_contributions: Mapping[str, Any] | None = None
    reason: str = ""
    warnings: tuple[str, ...] = ()
    prompt_safe: bool = True


class CacheEstimator(Protocol):
    def estimate(
        self,
        original_requests: Sequence[Any],
        rewritten_requests: Sequence[Any],
        candidate: Any | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> CacheEstimateReport:
        ...


class PrefixTreeEstimator:
    """Prompt-safe offline estimator based on planner prefix-tree metadata."""

    name = "prefix_tree"

    def estimate(
        self,
        original_requests: Sequence[Any],
        rewritten_requests: Sequence[Any],
        candidate: Any | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> CacheEstimateReport:
        del original_requests, rewritten_requests, candidate
        context = context or {}
        compile_result = context.get("compile_result")
        plan = context.get("plan")
        if not isinstance(compile_result, CompileResult) or not isinstance(plan, PrefixPlan):
            return CacheEstimateReport(
                estimator_name=self.name,
                estimator_available=False,
                reason="missing_compile_result_or_plan",
            )

        blocks_by_id = {block.block_id: block for block in compile_result.blocks}
        cacheable_ids = tuple(block_id for block_id in plan.cacheable_prefix_blocks if block_id in blocks_by_id)
        cacheable_set = set(cacheable_ids)
        original_prefix_chars = _leading_cacheable_chars(plan.original_order, cacheable_set, blocks_by_id)
        rewritten_prefix_chars = _leading_cacheable_chars(plan.new_order, cacheable_set, blocks_by_id)
        raw_delta = rewritten_prefix_chars - original_prefix_chars
        cache_report = plan.cache_gain_report or {}
        estimated_gain = max(0, raw_delta)
        if plan.moved_blocks and cache_report.get("estimated_cache_gain") is not None:
            estimated_gain = max(estimated_gain, float(cache_report.get("estimated_cache_gain") or 0.0))

        node_contributions = (
            cache_report.get("node_cache_contributions")
            if isinstance(cache_report.get("node_cache_contributions"), Mapping)
            else None
        )
        placement_contributions = (
            cache_report.get("placement_cache_contributions")
            if isinstance(cache_report.get("placement_cache_contributions"), Mapping)
            else None
        )
        if placement_contributions is None and plan.placements:
            placement_contributions = {
                placement.placement_id: placement.cache_contribution for placement in plan.placements
            }

        original_total = _order_chars(plan.original_order, blocks_by_id)
        rewritten_total = _order_chars(plan.new_order, blocks_by_id)
        return CacheEstimateReport(
            estimator_name=self.name,
            estimator_available=True,
            original_cacheable_tokens_or_chars=original_prefix_chars,
            rewritten_cacheable_tokens_or_chars=rewritten_prefix_chars,
            cached_tokens_delta=int(estimated_gain),
            cache_hit_ratio_before=_safe_ratio(original_prefix_chars, original_total),
            cache_hit_ratio_after=_safe_ratio(rewritten_prefix_chars, rewritten_total),
            estimated_cache_gain=estimated_gain,
            global_prefix_contribution=float(cache_report.get("global_prefix_tokens") or 0.0),
            subgroup_prefix_contribution=float(cache_report.get("subgroup_prefix_tokens") or 0.0),
            lcp_contribution=float(cache_report.get("longest_common_prefix_tokens") or 0.0),
            node_contributions=node_contributions,
            placement_contributions=placement_contributions,
            reason="prefix_tree_cache_gain_report" if cache_report else "prefix_tree_prefix_chars",
        )


class CacheHitProxyEstimator:
    """Adapter around cache_hit_proxy/cache_estimator.py without modifying that package."""

    name = "cache_hit_proxy"

    def __init__(
        self,
        *,
        estimator_module_path: str | Path | None = None,
        fallback_estimator: CacheEstimator | None = None,
        model_name: str = "autogen-prefix-tree-offline",
        block_size: int = 1,
        max_history_requests: int = 256,
        history_requests: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.estimator_module_path = Path(estimator_module_path) if estimator_module_path is not None else _default_proxy_path()
        self.fallback_estimator = fallback_estimator
        self.model_name = model_name
        self.block_size = max(int(block_size), 1)
        self.max_history_requests = max(int(max_history_requests), 0)
        self._history: list[dict[str, Any]] = [dict(item) for item in history_requests]
        self._estimate_func: Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]] | None = None
        self._load_error: str | None = None
        self.calls = 0

    @property
    def history_size(self) -> int:
        return len(self._history)

    def estimate(
        self,
        original_requests: Sequence[Any],
        rewritten_requests: Sequence[Any],
        candidate: Any | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> CacheEstimateReport:
        self.calls += 1
        estimate_func = self._load_estimate_func()
        if estimate_func is None:
            return self._fallback_or_unavailable(
                reason=self._load_error or "cache_hit_proxy_unavailable",
                original_requests=original_requests,
                rewritten_requests=rewritten_requests,
                candidate=candidate,
                context=context,
            )

        try:
            original_record, rewritten_record = self._records_for(
                original_requests=original_requests,
                rewritten_requests=rewritten_requests,
                context=context,
            )
            before = estimate_func(original_record, list(self._history))
            after = estimate_func(rewritten_record, list(self._history))
            before_cached = int(before.get("estimated_cached_tokens") or 0)
            after_cached = int(after.get("estimated_cached_tokens") or 0)
            delta = after_cached - before_cached
            return CacheEstimateReport(
                estimator_name=self.name,
                estimator_available=True,
                original_cacheable_tokens_or_chars=before_cached,
                rewritten_cacheable_tokens_or_chars=after_cached,
                cached_tokens_delta=delta,
                cache_hit_ratio_before=_float_or_none(before.get("estimated_cache_hit_rate")),
                cache_hit_ratio_after=_float_or_none(after.get("estimated_cache_hit_rate")),
                estimated_cache_gain=float(max(0, delta)),
                reason="cache_hit_proxy_estimate_cache_hit",
                warnings=(
                    "offline_hash_token_units",
                    "not_provider_reported_cached_tokens",
                    f"before_strategy:{before.get('match_strategy')}",
                    f"after_strategy:{after.get('match_strategy')}",
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return self._fallback_or_unavailable(
                reason=f"cache_hit_proxy_exception:{type(exc).__name__}",
                original_requests=original_requests,
                rewritten_requests=rewritten_requests,
                candidate=candidate,
                context=context,
            )

    def observe(
        self,
        original_requests: Sequence[Any],
        rewritten_requests: Sequence[Any],
        *,
        accepted: bool,
        candidate: Any | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        del candidate
        try:
            original_record, rewritten_record = self._records_for(
                original_requests=original_requests,
                rewritten_requests=rewritten_requests,
                context=context,
            )
        except Exception:
            return
        self._history.append(rewritten_record if accepted else original_record)
        if self.max_history_requests > 0 and len(self._history) > self.max_history_requests:
            self._history = self._history[-self.max_history_requests :]

    def _load_estimate_func(self) -> Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]] | None:
        if self._estimate_func is not None:
            return self._estimate_func
        if self._load_error is not None:
            return None
        if not self.estimator_module_path.exists():
            self._load_error = f"cache_hit_proxy_missing:{self.estimator_module_path.name}"
            return None
        try:
            spec = importlib.util.spec_from_file_location(
                "_autogen_prefix_tree_cache_hit_proxy_estimator",
                self.estimator_module_path,
            )
            if spec is None or spec.loader is None:
                self._load_error = "cache_hit_proxy_import_spec_unavailable"
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            estimate_func = getattr(module, "estimate_cache_hit", None)
            if not callable(estimate_func):
                self._load_error = "cache_hit_proxy_estimate_cache_hit_missing"
                return None
            self._estimate_func = estimate_func
            return self._estimate_func
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"cache_hit_proxy_import_exception:{type(exc).__name__}"
            return None

    def _fallback_or_unavailable(
        self,
        *,
        reason: str,
        original_requests: Sequence[Any],
        rewritten_requests: Sequence[Any],
        candidate: Any | None,
        context: Mapping[str, Any] | None,
    ) -> CacheEstimateReport:
        if self.fallback_estimator is None:
            return CacheEstimateReport(
                estimator_name=self.name,
                estimator_available=False,
                reason=reason,
                warnings=("cache_hit_proxy_adapter_failed",),
            )
        fallback_report = self.fallback_estimator.estimate(
            original_requests,
            rewritten_requests,
            candidate=candidate,
            context=context,
        )
        return replace(
            fallback_report,
            used_fallback=True,
            reason=f"{reason};fallback:{fallback_report.estimator_name}",
            warnings=tuple((*fallback_report.warnings, "cache_hit_proxy_adapter_failed")),
        )

    def _records_for(
        self,
        *,
        original_requests: Sequence[Any],
        rewritten_requests: Sequence[Any],
        context: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        original_record = _request_mapping_or_none(original_requests)
        rewritten_record = _request_mapping_or_none(rewritten_requests)
        if original_record is not None and rewritten_record is not None:
            return original_record, rewritten_record

        context = context or {}
        compile_result = context.get("compile_result")
        plan = context.get("plan")
        if not isinstance(compile_result, CompileResult) or not isinstance(plan, PrefixPlan):
            raise ValueError("compile_result_and_plan_required")
        blocks_by_id = {block.block_id: block for block in compile_result.blocks}
        session_id = str(context.get("session_id") or compile_result.session_id)
        request_id_prefix = stable_hash(
            {
                "session_id": session_id,
                "candidate_id": getattr(context.get("candidate"), "candidate_id", None),
                "original_order": plan.original_order,
                "new_order": plan.new_order,
            }
        )[:16]
        return (
            self._record_from_order(
                order=plan.original_order,
                blocks_by_id=blocks_by_id,
                request_id=f"{session_id}:{request_id_prefix}:original",
            ),
            self._record_from_order(
                order=plan.new_order,
                blocks_by_id=blocks_by_id,
                request_id=f"{session_id}:{request_id_prefix}:rewritten",
            ),
        )

    def _record_from_order(
        self,
        *,
        order: Sequence[str],
        blocks_by_id: Mapping[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        token_ids: list[int] = []
        persisted_units: list[list[int]] = []
        for block_id in order:
            block = blocks_by_id.get(block_id)
            if block is None:
                continue
            token_ids.extend(_block_token_ids(block))
            if token_ids:
                persisted_units.append(list(token_ids))
        return {
            "request_id": request_id,
            "model": self.model_name,
            "token_ids": token_ids,
            "persisted_prefix_units_tokens": persisted_units,
            "cache_block_size": self.block_size,
            "cache_estimation_input_tokens": len(token_ids),
            "_cache_unit_source": "deepseek_prompt_encoding",
        }


class ProviderTelemetryEstimator:
    """Reserved interface for future provider-reported cached-token metrics."""

    name = "provider_telemetry"

    def __init__(self, *, fallback_estimator: CacheEstimator | None = None) -> None:
        self.fallback_estimator = fallback_estimator

    def estimate(
        self,
        original_requests: Sequence[Any],
        rewritten_requests: Sequence[Any],
        candidate: Any | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> CacheEstimateReport:
        if self.fallback_estimator is not None:
            fallback_report = self.fallback_estimator.estimate(
                original_requests,
                rewritten_requests,
                candidate=candidate,
                context=context,
            )
            return replace(
                fallback_report,
                used_fallback=True,
                reason=f"provider_telemetry_not_configured;fallback:{fallback_report.estimator_name}",
                warnings=tuple((*fallback_report.warnings, "provider_telemetry_placeholder")),
            )
        return CacheEstimateReport(
            estimator_name=self.name,
            estimator_available=False,
            reason="provider_telemetry_not_configured",
            warnings=("provider_telemetry_placeholder",),
        )


def resolve_cache_estimator(cache_estimator: str | CacheEstimator | None = None) -> CacheEstimator:
    if cache_estimator is None:
        return PrefixTreeEstimator()
    if isinstance(cache_estimator, str):
        key = cache_estimator.strip().lower().replace("-", "_")
        if key in {"", "prefix_tree", "prefixtree"}:
            return PrefixTreeEstimator()
        if key in {"cache_hit_proxy", "cache_proxy"}:
            return CacheHitProxyEstimator(fallback_estimator=PrefixTreeEstimator())
        if key in {"provider_telemetry", "provider"}:
            return ProviderTelemetryEstimator(fallback_estimator=PrefixTreeEstimator())
        raise ValueError(f"Unsupported cache_estimator: {cache_estimator}")
    if not callable(getattr(cache_estimator, "estimate", None)):
        raise TypeError("cache_estimator must be a string or expose estimate(...)")
    return cache_estimator


def cache_estimator_config_name(cache_estimator: str | CacheEstimator | None) -> str:
    if cache_estimator is None:
        return "prefix_tree"
    if isinstance(cache_estimator, str):
        return cache_estimator.strip().lower().replace("-", "_") or "prefix_tree"
    return getattr(cache_estimator, "name", type(cache_estimator).__name__)


def observe_cache_estimator(
    cache_estimator: CacheEstimator,
    original_requests: Sequence[Any],
    rewritten_requests: Sequence[Any],
    *,
    accepted: bool,
    candidate: Any | None = None,
    context: Mapping[str, Any] | None = None,
) -> None:
    observe = getattr(cache_estimator, "observe", None)
    if not callable(observe):
        return
    try:
        observe(
            original_requests,
            rewritten_requests,
            accepted=accepted,
            candidate=candidate,
            context=context,
        )
    except Exception:
        return


def cache_gate_value(report: CacheEstimateReport | None) -> float:
    if report is None:
        return 0.0
    if report.cached_tokens_delta is not None:
        return float(report.cached_tokens_delta)
    return float(report.estimated_cache_gain)


def _default_proxy_path() -> Path:
    return Path(__file__).resolve().parents[2] / "cache_hit_proxy" / "cache_estimator.py"


def _leading_cacheable_chars(order: Sequence[str], cacheable_set: set[str], blocks_by_id: Mapping[str, Any]) -> int:
    chars = 0
    for block_id in order:
        if block_id not in cacheable_set:
            break
        chars += len(getattr(blocks_by_id[block_id], "rendered_text", None) or "")
    return chars


def _order_chars(order: Sequence[str], blocks_by_id: Mapping[str, Any]) -> int:
    return sum(len(getattr(blocks_by_id[block_id], "rendered_text", None) or "") for block_id in order if block_id in blocks_by_id)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _request_mapping_or_none(values: Sequence[Any]) -> dict[str, Any] | None:
    if len(values) != 1:
        return None
    value = values[0]
    if isinstance(value, Mapping) and isinstance(value.get("token_ids"), list):
        return dict(value)
    return None


def _block_token_ids(block: Any) -> list[int]:
    content_hash = str(getattr(block, "content_hash", None) or getattr(block, "block_hash", None) or "")
    if not content_hash:
        content_hash = stable_hash(getattr(block, "block_id", "unknown"))
    token_len = int(getattr(block, "token_len", 0) or 0)
    if token_len <= 0:
        token_len = max(1, len(str(getattr(block, "rendered_text", "") or "")) // 4)
    base = int(hashlib.sha256(content_hash.encode("utf-8")).hexdigest()[:12], 16)
    return [base + offset for offset in range(max(1, token_len))]
