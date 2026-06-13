from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from autogen_core.models import SystemMessage

from .compiler import LocalPromptCompiler
from .ir import stable_hash
from .planner import HierarchicalPrefixPlanner
from .planner_ranker_baseline import PlannerRankerModel
from .project_api_config import load_project_provider_config, normalize_deepseek_model
from .telemetry import serialize_prefix_tree_candidate
from .utility_dataset_builder import (
    EXPANDED_STRATEGIES,
    _candidate_strategy,
    _expanded_candidates_for_task,
    _messages_to_prompt,
)
from .utility_oracles import OracleResult, oracle_for_task
from .utility_validator_baseline import UtilityValidatorModel, runtime_feature_row_from_gate_inputs
from .validator import CacheUtilityValidator

PERF_SPLIT = "perf_eval_dspro"
SOURCE_ORDER = ("humaneval", "synthetic_agent", "format_protocol", "role_privacy", "history_state")
KNOWN_CACHE_KEYS = (
    "cached_tokens",
    "cached_prompt_tokens",
    "prompt_cache_hit_tokens",
    "cache_hit_tokens",
    "input_cached_tokens",
)
KNOWN_CACHE_DETAIL_KEYS = ("prompt_tokens_details", "input_token_details", "input_tokens_details")

GLOBAL_SHARED_BLOCK = """GLOBAL_PERF_EVAL_CONTEXT_START
This is the held-out DS Pro performance evaluation workload for semantic prefix caching in multi-agent prompts.
All tasks must preserve role boundaries, tool permission boundaries, private memory boundaries, output protocol, and state order.
The optimization framework may move only repeated shared context into a common prefix while preserving task-specific and agent-specific constraints.
The experiment compares original prompt ordering against a prefix-tree optimized ordering using provider token telemetry.
GLOBAL_PERF_EVAL_CONTEXT_END"""

TEAM_POLICY_BLOCK = """TEAM_POLICY_START
Preserve the latest task instruction, do not expose private memory, do not assume a role that belongs to another agent, use only allowed tools, keep required output schemas exact, and preserve the documented event order.
Return only the requested JSON object for the evaluator harness.
TEAM_POLICY_END"""

SHARED_OUTPUT_BLOCK = """OUTPUT_PROTOCOL_START
The evaluator consumes strict JSON. Do not include markdown fences, commentary, chain-of-thought, prompt text, API keys, or private content in the final response.
OUTPUT_PROTOCOL_END"""


class ExperimentAbort(RuntimeError):
    pass


class DsProClient:
    def __init__(
        self,
        *,
        repo_root: Path,
        model: str | None,
        max_api_calls: int,
        confirm_cost_aware: bool,
        max_output_tokens: int,
        timeout_seconds: float,
        price_input_per_million: float,
        price_cached_input_per_million: float,
        price_output_per_million: float,
        project_config_path: str | Path | None,
    ) -> None:
        if not confirm_cost_aware:
            raise ValueError("--confirm-cost-aware is required before any DS Pro API call")
        if max_api_calls <= 0:
            raise ValueError("--max-api-calls must be positive before any DS Pro API call")
        config = load_project_provider_config(cwd=repo_root, config_path=project_config_path)
        if config is None:
            raise ValueError("DeepSeek config/API key was not found; no API call was made")
        effective_model = normalize_deepseek_model(model or config.model)
        if not _is_pro_model(effective_model) or _is_flash_model(effective_model):
            raise ValueError(f"DS Pro experiment requires a Pro model and forbids Flash; got {effective_model!r}")
        self.config = config
        self.model = effective_model
        self.max_api_calls = max_api_calls
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.price_input_per_million = price_input_per_million
        self.price_cached_input_per_million = price_cached_input_per_million
        self.price_output_per_million = price_output_per_million
        self.calls: list[dict[str, Any]] = []

    @property
    def calls_made(self) -> int:
        return len(self.calls)

    def call(self, *, task: Mapping[str, Any], group: str, prompt: str) -> dict[str, Any]:
        if self.calls_made >= self.max_api_calls:
            raise ExperimentAbort(f"max api calls reached: {self.max_api_calls}")
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": _prompt_execution_user_content(task=task, prompt=prompt)}],
            "temperature": 0,
            "stream": False,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        started = time.perf_counter()
        response = _post_chat_completion(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            body=body,
            timeout_seconds=self.timeout_seconds,
        )
        latency = time.perf_counter() - started
        usage = _usage_from_response(response)
        if not usage["cached_tokens_present"]:
            raise ExperimentAbort(f"cached_tokens missing from provider usage at call {self.calls_made + 1}")
        content = _response_content(response)
        report = {
            "call_index": self.calls_made + 1,
            "task_id": task.get("task_id"),
            "dataset_source": task.get("dataset_source"),
            "group": group,
            "model": self.model,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "cached_tokens": usage["cached_tokens"],
            "cache_miss_tokens": usage["cache_miss_tokens"],
            "cached_tokens_present": usage["cached_tokens_present"],
            "latency_seconds": latency,
            "estimated_cost_usd": self._estimated_cost_usd(usage),
            "provider_cost_usd": usage.get("provider_cost_usd"),
            "response_hash": stable_hash(content),
            "_local_response_content": content,
        }
        self.calls.append(report)
        return report

    def cost_summary(self) -> dict[str, Any]:
        return {
            "api_call_count": self.calls_made,
            "max_api_calls": self.max_api_calls,
            "model": self.model,
            "model_is_pro": _is_pro_model(self.model),
            "model_is_flash": _is_flash_model(self.model),
            "provider": self.config.provider,
            "base_url": self.config.base_url,
            "config_source": "env" if self.config.config_path is None else "config/config.txt",
            "api_key_stored": False,
            "total_input_tokens": sum(_int(call.get("input_tokens")) for call in self.calls),
            "total_output_tokens": sum(_int(call.get("output_tokens")) for call in self.calls),
            "total_cached_tokens": sum(_int(call.get("cached_tokens")) for call in self.calls),
            "total_cache_miss_tokens": sum(_int(call.get("cache_miss_tokens")) for call in self.calls),
            "total_tokens": sum(_int(call.get("total_tokens")) for call in self.calls),
            "estimated_cost_usd": sum(float(call.get("estimated_cost_usd") or 0.0) for call in self.calls),
            "provider_cost_usd": sum(float(call.get("provider_cost_usd") or 0.0) for call in self.calls),
            "latency_seconds_total": sum(float(call.get("latency_seconds") or 0.0) for call in self.calls),
            "latency_seconds_avg": _safe_div(
                sum(float(call.get("latency_seconds") or 0.0) for call in self.calls),
                len(self.calls),
            ),
            "price_input_per_million": self.price_input_per_million,
            "price_cached_input_per_million": self.price_cached_input_per_million,
            "price_output_per_million": self.price_output_per_million,
        }

    def _estimated_cost_usd(self, usage: Mapping[str, Any]) -> float:
        if usage.get("provider_cost_usd") is not None:
            return float(usage["provider_cost_usd"])
        cached = min(_int(usage.get("cached_tokens")), _int(usage.get("input_tokens")))
        uncached = max(0, _int(usage.get("input_tokens")) - cached)
        output = _int(usage.get("output_tokens"))
        return (
            uncached / 1_000_000 * self.price_input_per_million
            + cached / 1_000_000 * self.price_cached_input_per_million
            + output / 1_000_000 * self.price_output_per_million
        )


def run_dspro_performance_eval(
    *,
    repo_root: str | Path = ".",
    tasks_path: str | Path = "datasets/utility_validator/tasks/perf_eval_dspro/utility_tasks.jsonl",
    output_dir: str | Path = "artifacts/performance_eval/dspro_main",
    backend: str = "dsapi",
    confirm_cost_aware: bool = False,
    max_api_calls: int = 240,
    checkpoint_api_calls: int = 50,
    model: str | None = None,
    max_output_tokens: int = 160,
    timeout_seconds: float = 120.0,
    max_estimated_cost_usd: float = 5.0,
    min_total_input_tokens: int = 100_000,
    max_total_input_tokens: int = 500_000,
    dry_run: bool = False,
    project_config_path: str | Path | None = None,
    cache_isolation_tag: str | None = None,
    force_front_loaded_common_prefix: bool = False,
) -> dict[str, Any]:
    if backend != "dsapi":
        raise ValueError("This experiment must be run with --backend dsapi")
    repo = Path(repo_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)

    tasks = _read_jsonl(repo / tasks_path)
    _validate_task_set(tasks)
    prepared = _prepare_prompts(repo_root=repo, tasks=tasks, cache_isolation_tag=cache_isolation_tag)
    prompt_budget = _prompt_budget(prepared)
    if prompt_budget["estimated_total_input_tokens"] > max_total_input_tokens:
        raise ExperimentAbort(
            f"estimated input tokens exceed budget: {prompt_budget['estimated_total_input_tokens']} > {max_total_input_tokens}"
        )

    utility_model = _load_optional_utility_model(repo)
    planner_ranker = _load_optional_planner_ranker(repo)
    optimized = _choose_optimized_prompts(
        prepared=prepared,
        utility_model=utility_model,
        planner_ranker=planner_ranker,
        force_front_loaded_common_prefix=force_front_loaded_common_prefix,
    )
    if dry_run:
        summary = {
            "schema_version": "dspro-performance-eval-dry-run-v1",
            "task_count": len(tasks),
            "source_counts": dict(Counter(str(task["dataset_source"]) for task in tasks)),
            "prompt_budget": prompt_budget,
        "utility_model_loaded": utility_model is not None,
        "planner_ranker_loaded": planner_ranker is not None,
        "force_front_loaded_common_prefix": force_front_loaded_common_prefix,
        "optimized_prompt_count": len(optimized),
        "ds_api_called": False,
        }
        _write_json(output / "dry_run_summary.json", summary)
        return summary

    client = DsProClient(
        repo_root=repo,
        model=model,
        max_api_calls=max_api_calls,
        confirm_cost_aware=confirm_cost_aware,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        price_input_per_million=1.74,
        price_cached_input_per_million=0.174,
        price_output_per_million=3.48,
        project_config_path=project_config_path,
    )
    baseline_rows: list[dict[str, Any]] = []
    optimized_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    checkpoint_errors: list[str] = []

    try:
        for item in prepared:
            baseline_call = client.call(task=item["task"], group="baseline", prompt=item["baseline_prompt"])
            baseline_row = _result_row(
                task=item["task"],
                group="baseline",
                prompt=item["baseline_prompt"],
                call=baseline_call,
                candidate=None,
                oracle_spec=item["oracle_spec"],
                validation_summary=None,
            )
            baseline_rows.append(baseline_row)
            _write_jsonl(output / "baseline_results.jsonl", baseline_rows)
            _checkpoint_if_needed(
                output=output,
                client=client,
                baseline_rows=baseline_rows,
                optimized_rows=optimized_rows,
                checkpoints=checkpoints,
                checkpoint_errors=checkpoint_errors,
                checkpoint_api_calls=checkpoint_api_calls,
            )
            _abort_if_unhealthy(client=client, rows=baseline_rows + optimized_rows, max_estimated_cost_usd=max_estimated_cost_usd)

            chosen = optimized[str(item["task"]["task_id"])]
            optimized_call = client.call(task=item["task"], group="optimized", prompt=chosen["prompt"])
            optimized_row = _result_row(
                task=item["task"],
                group="optimized",
                prompt=chosen["prompt"],
                call=optimized_call,
                candidate=chosen,
                oracle_spec=item["oracle_spec"],
                validation_summary=chosen.get("validation_summary"),
            )
            optimized_rows.append(optimized_row)
            _write_jsonl(output / "optimized_results.jsonl", optimized_rows)
            _checkpoint_if_needed(
                output=output,
                client=client,
                baseline_rows=baseline_rows,
                optimized_rows=optimized_rows,
                checkpoints=checkpoints,
                checkpoint_errors=checkpoint_errors,
                checkpoint_api_calls=checkpoint_api_calls,
            )
            _abort_if_unhealthy(client=client, rows=baseline_rows + optimized_rows, max_estimated_cost_usd=max_estimated_cost_usd)
    finally:
        for call in client.calls:
            call.pop("_local_response_content", None)

    summary = _write_reports(
        output=output,
        tasks=tasks,
        prepared=prepared,
        baseline_rows=baseline_rows,
        optimized_rows=optimized_rows,
        client=client,
        checkpoints=checkpoints,
        checkpoint_errors=checkpoint_errors,
        prompt_budget=prompt_budget,
        min_total_input_tokens=min_total_input_tokens,
        max_total_input_tokens=max_total_input_tokens,
        utility_model_loaded=utility_model is not None,
        planner_ranker_loaded=planner_ranker is not None,
        cache_isolation_tag=cache_isolation_tag,
        force_front_loaded_common_prefix=force_front_loaded_common_prefix,
    )
    _scan_artifacts_for_sensitive_text(output)
    return summary


def _prepare_prompts(
    *,
    repo_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    cache_isolation_tag: str | None = None,
) -> list[dict[str, Any]]:
    humaneval_rows = _load_humaneval_rows(repo_root)
    prepared: list[dict[str, Any]] = []
    for task in tasks:
        oracle_spec = _runtime_oracle_spec(task)
        if task["dataset_source"] == "humaneval":
            source_id = str(task["source_metadata"]["source_task_id"])
            source_row = humaneval_rows[source_id]
            task_body = _humaneval_task_body(source_row)
        else:
            task_body = _synthetic_task_body(task, oracle_spec)
        baseline_prompt = _baseline_prompt(
            task=task,
            task_body=task_body,
            oracle_spec=oracle_spec,
            cache_isolation_tag=cache_isolation_tag,
        )
        messages = (SystemMessage(content=baseline_prompt),)
        prepared.append(
            {
                "task": task,
                "oracle_spec": oracle_spec,
                "baseline_prompt": baseline_prompt,
                "messages": messages,
            }
        )
    return prepared


def _choose_optimized_prompts(
    *,
    prepared: Sequence[Mapping[str, Any]],
    utility_model: UtilityValidatorModel | None,
    planner_ranker: PlannerRankerModel | None,
    force_front_loaded_common_prefix: bool = False,
) -> dict[str, dict[str, Any]]:
    compiler = LocalPromptCompiler()
    planner = HierarchicalPrefixPlanner()
    validator = CacheUtilityValidator(
        utility_model_validator=utility_model,
        utility_model_mode="gate" if utility_model is not None else "disabled",
        utility_model_min_confidence=0.55,
    )
    chosen_by_task: dict[str, dict[str, Any]] = {}
    session_id = "perf-eval-dspro-prefix-tree"
    for item in prepared:
        task = item["task"]
        compile_result = compiler.compile(item["messages"], session_id=session_id)
        candidates = _expanded_candidates_for_task(
            planner=planner,
            compile_result=compile_result,
            session_id=session_id,
            strategies=EXPANDED_STRATEGIES,
        )
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            plan = planner._plan_from_candidate(compile_result, candidate)
            rewritten = _messages_for_plan(planner=planner, compile_result=compile_result, plan=plan)
            validation = validator.validate(
                original_messages=compile_result.messages,
                rewritten_messages=rewritten,
                compile_result=compile_result,
                plan=plan,
            )
            feature_row = runtime_feature_row_from_gate_inputs(
                original_messages=compile_result.messages,
                rewritten_messages=rewritten,
                compile_result=compile_result,
                plan=plan,
                cache_utility_estimate=getattr(validation, "utility_estimate", None),
            )
            feature_row.update(
                {
                    "dataset_source": task.get("dataset_source"),
                    "scenario_type": task.get("scenario_type"),
                    "candidate_strategy": _candidate_strategy(candidate),
                }
            )
            ranker_prediction = (
                planner_ranker.predict(feature_row)
                if planner_ranker is not None
                else {
                    "priority_score": float(getattr(plan, "estimated_cache_gain", 0.0) or 0.0),
                    "reason": "planner_score_fallback_no_ranker_model",
                    "model_name": None,
                }
            )
            scored.append(
                {
                    "candidate": candidate,
                    "plan": plan,
                    "prompt": _messages_to_prompt(rewritten),
                    "validation": validation,
                    "ranker_prediction": ranker_prediction,
                    "feature_row_hash": stable_hash(feature_row),
                }
            )
        accepted = [row for row in scored if getattr(row["validation"], "applied", False)]
        pool = accepted or scored
        if pool:
            best = sorted(
                pool,
                key=lambda row: (
                    -float(row["ranker_prediction"].get("priority_score") or 0.0),
                    -float(getattr(row["plan"], "estimated_cache_gain", 0.0) or 0.0),
                    str(getattr(row["candidate"], "candidate_id", "")),
                ),
            )[0]
            prompt = best["prompt"]
            if force_front_loaded_common_prefix:
                prompt = _front_load_common_prefix(
                    prompt,
                    cache_isolation_tag=str(item["task"].get("source_metadata", {}).get("cache_isolation_tag") or ""),
                    task_id=str(item["task"].get("task_id") or ""),
                )
            candidate = best["candidate"]
            plan = best["plan"]
            chosen_by_task[str(task["task_id"])] = {
                "prompt": prompt,
                "prompt_hash": stable_hash(prompt),
                "candidate_id": getattr(candidate, "candidate_id", None),
                "candidate_strategy": _candidate_strategy(candidate),
                "estimated_cache_gain": float(getattr(plan, "estimated_cache_gain", 0.0) or 0.0),
                "ranker_prediction": _safe_public_mapping(best["ranker_prediction"]),
                "validation_summary": _validation_summary(best["validation"]),
                "candidate_summary": serialize_prefix_tree_candidate(candidate, include_text=False),
                "feature_row_hash": best["feature_row_hash"],
            }
        else:
            prompt = item["baseline_prompt"]
            chosen_by_task[str(task["task_id"])] = {
                "prompt": prompt,
                "prompt_hash": stable_hash(prompt),
                "candidate_id": None,
                "candidate_strategy": "fallback_no_candidate",
                "estimated_cache_gain": 0.0,
                "ranker_prediction": {"reason": "no_candidate"},
                "validation_summary": {"applied": False, "fallback": True, "reason": "no_candidate"},
                "candidate_summary": None,
            }
        planner.observe(compile_result, session_id=session_id)
    return chosen_by_task


def _messages_for_plan(*, planner: HierarchicalPrefixPlanner, compile_result: Any, plan: Any) -> tuple[SystemMessage, ...]:
    candidate = getattr(plan, "prefix_tree_candidate", None)
    if candidate is not None and getattr(candidate, "block_text_by_id", None):
        agent_id = next(iter(candidate.agent_block_orders.keys()), None)
        if agent_id:
            return (SystemMessage(content=planner.materialize_prompt(candidate, agent_id)),)
    ordered = [block for block_id in plan.new_order for block in compile_result.blocks if block.block_id == block_id]
    return (SystemMessage(content="".join(str(getattr(block, "rendered_text", "")) for block in ordered)),)


def _front_load_common_prefix(prompt: str, *, cache_isolation_tag: str = "", task_id: str = "") -> str:
    common_blocks = _shared_common_blocks(cache_isolation_tag)
    remainder = prompt
    for block in common_blocks:
        remainder = remainder.replace(block, "", 1)
    isolation = (
        f"CACHE_ISOLATION_TASK_TAG:{cache_isolation_tag}:{task_id}"
        if cache_isolation_tag and task_id
        else ""
    )
    blocks = (*common_blocks, isolation, remainder.strip()) if isolation else (*common_blocks, remainder.strip())
    return "\n\n".join(block for block in blocks if block)


def _result_row(
    *,
    task: Mapping[str, Any],
    group: str,
    prompt: str,
    call: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    oracle_spec: Mapping[str, Any],
    validation_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    content = str(call.get("_local_response_content") or "")
    parsed = _parse_model_json(content)
    oracle = oracle_for_task(task, oracle_spec)
    oracle_result = oracle.evaluate(task, parsed if parsed is not None else content, parsed)
    success = bool(oracle_result.passed)
    return {
        "schema_version": "dspro-performance-result-v1",
        "task_id": task.get("task_id"),
        "dataset_source": task.get("dataset_source"),
        "scenario_type": task.get("scenario_type"),
        "eval_split": PERF_SPLIT,
        "group": group,
        "model": call.get("model"),
        "input_tokens": call.get("input_tokens"),
        "output_tokens": call.get("output_tokens"),
        "cached_tokens": call.get("cached_tokens"),
        "cache_miss_tokens": call.get("cache_miss_tokens"),
        "latency": call.get("latency_seconds"),
        "latency_seconds": call.get("latency_seconds"),
        "estimated_cost": call.get("estimated_cost_usd"),
        "estimated_cost_usd": call.get("estimated_cost_usd"),
        "success": success,
        "failure_type": oracle_result.failure_type if not success else "none",
        "oracle_reason": oracle_result.reason,
        "oracle_metrics": dict(oracle_result.metrics),
        "call_index": call.get("call_index"),
        "response_hash": call.get("response_hash"),
        "response_json_keys": sorted(parsed.keys()) if isinstance(parsed, Mapping) else [],
        "prompt_hash": stable_hash(prompt),
        "prompt_text_included": False,
        "candidate_id": (candidate or {}).get("candidate_id"),
        "candidate_strategy": (candidate or {}).get("candidate_strategy") or ("baseline" if group == "baseline" else None),
        "estimated_cache_gain": (candidate or {}).get("estimated_cache_gain"),
        "validation_summary": validation_summary,
        "ranker_prediction": (candidate or {}).get("ranker_prediction"),
    }


def _write_reports(
    *,
    output: Path,
    tasks: Sequence[Mapping[str, Any]],
    prepared: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    optimized_rows: Sequence[Mapping[str, Any]],
    client: DsProClient,
    checkpoints: Sequence[Mapping[str, Any]],
    checkpoint_errors: Sequence[str],
    prompt_budget: Mapping[str, Any],
    min_total_input_tokens: int,
    max_total_input_tokens: int,
    utility_model_loaded: bool,
    planner_ranker_loaded: bool,
    cache_isolation_tag: str | None,
    force_front_loaded_common_prefix: bool,
) -> dict[str, Any]:
    comparison = _comparison_summary(baseline_rows, optimized_rows)
    per_source = _per_source_metrics(baseline_rows, optimized_rows)
    cache_report = _cache_report(baseline_rows, optimized_rows, per_source)
    utility_report = _utility_report(baseline_rows, optimized_rows, per_source)
    cost_summary = client.cost_summary()
    cost_summary.update(
        {
            "input_token_target_min": min_total_input_tokens,
            "input_token_target_max": max_total_input_tokens,
            "prompt_budget_estimate": dict(prompt_budget),
        }
    )
    summary = {
        "schema_version": "dspro-performance-comparison-summary-v1",
        "created_at": _now_iso(),
        "task_count": len(tasks),
        "completed_task_count": min(len(baseline_rows), len(optimized_rows)),
        "actual_model": client.model,
        "backend": "dsapi",
        "confirm_cost_aware": True,
        "api_call_count": client.calls_made,
        "checkpoint_api_calls": 50,
        "checkpoint_count": len(checkpoints),
        "checkpoint_errors": list(checkpoint_errors),
        "aborted": len(baseline_rows) != len(tasks) or len(optimized_rows) != len(tasks),
        "utility_model_loaded": utility_model_loaded,
        "planner_ranker_loaded": planner_ranker_loaded,
        "cache_isolation_tag": cache_isolation_tag,
        "force_front_loaded_common_prefix": force_front_loaded_common_prefix,
        **comparison,
    }
    _write_json(output / "comparison_summary.json", summary)
    _write_json(output / "per_source_metrics.json", per_source)
    _write_json(output / "cache_report.json", cache_report)
    _write_json(output / "utility_report.json", utility_report)
    _write_json(output / "cost_summary.json", cost_summary)
    _write_json(output / "api_call_summary.json", {"calls": _public_calls(client.calls)})
    _write_experiment_report(
        output=output,
        summary=summary,
        per_source=per_source,
        cache_report=cache_report,
        utility_report=utility_report,
        cost_summary=cost_summary,
        prompt_budget=prompt_budget,
    )
    _write_figures(output / "figures", summary, per_source, cache_report, utility_report, cost_summary)
    return summary


def _comparison_summary(
    baseline_rows: Sequence[Mapping[str, Any]],
    optimized_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {str(row["task_id"]): row for row in baseline_rows}
    optimized_by_id = {str(row["task_id"]): row for row in optimized_rows}
    paired_ids = sorted(set(baseline_by_id) & set(optimized_by_id))
    baseline_input = sum(_int(baseline_by_id[task_id].get("input_tokens")) for task_id in paired_ids)
    baseline_cached = sum(_int(baseline_by_id[task_id].get("cached_tokens")) for task_id in paired_ids)
    optimized_input = sum(_int(optimized_by_id[task_id].get("input_tokens")) for task_id in paired_ids)
    optimized_cached = sum(_int(optimized_by_id[task_id].get("cached_tokens")) for task_id in paired_ids)
    baseline_success = sum(1 for task_id in paired_ids if baseline_by_id[task_id].get("success") is True)
    optimized_success = sum(1 for task_id in paired_ids if optimized_by_id[task_id].get("success") is True)
    preserved = sum(
        1
        for task_id in paired_ids
        if baseline_by_id[task_id].get("success") is True and optimized_by_id[task_id].get("success") is True
    )
    baseline_cache = _safe_div(baseline_cached, baseline_input)
    optimized_cache = _safe_div(optimized_cached, optimized_input)
    preservation = _safe_div(preserved, baseline_success)
    return {
        "paired_task_count": len(paired_ids),
        "baseline_cache_hit_rate": baseline_cache,
        "ours_cache_hit_rate": optimized_cache,
        "cache_hit_rate_lift": optimized_cache - baseline_cache,
        "cache_hit_rate_lift_percentage_points": (optimized_cache - baseline_cache) * 100.0,
        "baseline_success_rate": _safe_div(baseline_success, len(paired_ids)),
        "ours_success_rate": _safe_div(optimized_success, len(paired_ids)),
        "baseline_success_count": baseline_success,
        "ours_success_count": optimized_success,
        "utility_preservation_rate": preservation,
        "utility_drop": 1.0 - preservation if baseline_success else 0.0,
        "original_failed_count": len(paired_ids) - baseline_success,
    }


def _per_source_metrics(
    baseline_rows: Sequence[Mapping[str, Any]],
    optimized_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    baseline_by_id = {str(row["task_id"]): row for row in baseline_rows}
    optimized_by_id = {str(row["task_id"]): row for row in optimized_rows}
    for source in SOURCE_ORDER:
        ids = sorted(
            task_id
            for task_id, row in baseline_by_id.items()
            if row.get("dataset_source") == source and task_id in optimized_by_id
        )
        b_input = sum(_int(baseline_by_id[task_id].get("input_tokens")) for task_id in ids)
        b_cached = sum(_int(baseline_by_id[task_id].get("cached_tokens")) for task_id in ids)
        o_input = sum(_int(optimized_by_id[task_id].get("input_tokens")) for task_id in ids)
        o_cached = sum(_int(optimized_by_id[task_id].get("cached_tokens")) for task_id in ids)
        b_success = sum(1 for task_id in ids if baseline_by_id[task_id].get("success") is True)
        o_success = sum(1 for task_id in ids if optimized_by_id[task_id].get("success") is True)
        preserved = sum(
            1
            for task_id in ids
            if baseline_by_id[task_id].get("success") is True and optimized_by_id[task_id].get("success") is True
        )
        b_cache = _safe_div(b_cached, b_input)
        o_cache = _safe_div(o_cached, o_input)
        preservation = _safe_div(preserved, b_success)
        result[source] = {
            "task_count": len(ids),
            "baseline_input_tokens": b_input,
            "baseline_cached_tokens": b_cached,
            "ours_input_tokens": o_input,
            "ours_cached_tokens": o_cached,
            "baseline_cache_hit_rate": b_cache,
            "ours_cache_hit_rate": o_cache,
            "cache_hit_rate_lift": o_cache - b_cache,
            "baseline_success_rate": _safe_div(b_success, len(ids)),
            "ours_success_rate": _safe_div(o_success, len(ids)),
            "utility_preservation_rate": preservation,
            "utility_drop": 1.0 - preservation if b_success else 0.0,
            "original_failed_count": len(ids) - b_success,
        }
    return result


def _cache_report(
    baseline_rows: Sequence[Mapping[str, Any]],
    optimized_rows: Sequence[Mapping[str, Any]],
    per_source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "dspro-cache-report-v1",
        "overall": {
            key: value
            for key, value in _comparison_summary(baseline_rows, optimized_rows).items()
            if "cache" in key
        },
        "per_source": {source: {k: v for k, v in metrics.items() if "cache" in k} for source, metrics in per_source.items()},
    }


def _utility_report(
    baseline_rows: Sequence[Mapping[str, Any]],
    optimized_rows: Sequence[Mapping[str, Any]],
    per_source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "dspro-utility-report-v1",
        "overall": {
            key: value
            for key, value in _comparison_summary(baseline_rows, optimized_rows).items()
            if "success" in key or "utility" in key or "original_failed" in key
        },
        "failure_type_counts": {
            "baseline": dict(Counter(str(row.get("failure_type")) for row in baseline_rows)),
            "ours": dict(Counter(str(row.get("failure_type")) for row in optimized_rows)),
        },
        "per_source": {
            source: {k: v for k, v in metrics.items() if "success" in k or "utility" in k or "original_failed" in k}
            for source, metrics in per_source.items()
        },
    }


def _write_figures(
    figures_dir: Path,
    summary: Mapping[str, Any],
    per_source: Mapping[str, Any],
    cache_report: Mapping[str, Any],
    utility_report: Mapping[str, Any],
    cost_summary: Mapping[str, Any],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _write_figures_simple(figures_dir, summary, per_source)
        return

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(figures_dir / f"{name}.png", dpi=180)
        plt.savefig(figures_dir / f"{name}.svg")
        plt.close()

    plt.figure(figsize=(7, 4.5))
    vals = [summary["baseline_cache_hit_rate"], summary["ours_cache_hit_rate"]]
    plt.bar(["Baseline", "Ours"], vals, color=["#4c78a8", "#54a24b"])
    plt.ylabel("Cache hit rate")
    plt.ylim(0, max(1.0, max(vals) * 1.2))
    plt.title("Cache Hit Rate Comparison")
    plt.text(0.5, max(vals) * 1.05 if vals else 0.05, f"Lift: {summary['cache_hit_rate_lift_percentage_points']:.2f} pp", ha="center")
    save("cache_hit_rate_comparison")

    plt.figure(figsize=(8, 4.5))
    vals = [summary["baseline_success_rate"], summary["ours_success_rate"], summary["utility_preservation_rate"]]
    plt.bar(["Baseline success", "Ours success", "Preservation"], vals, color=["#4c78a8", "#54a24b", "#f58518"])
    plt.ylabel("Rate")
    plt.ylim(0, 1.05)
    plt.title("Utility Comparison")
    plt.text(1.2, 0.08, f"Utility drop: {summary['utility_drop']:.2%}", ha="center")
    save("utility_comparison")

    sources = list(SOURCE_ORDER)
    x = list(range(len(sources)))
    width = 0.36
    plt.figure(figsize=(9, 4.8))
    plt.bar([i - width / 2 for i in x], [per_source[s]["baseline_cache_hit_rate"] for s in sources], width, label="Baseline", color="#4c78a8")
    plt.bar([i + width / 2 for i in x], [per_source[s]["ours_cache_hit_rate"] for s in sources], width, label="Ours", color="#54a24b")
    plt.xticks(x, sources, rotation=20, ha="right")
    plt.ylabel("Cache hit rate")
    plt.ylim(0, 1.05)
    plt.title("Per-source Cache Hit Rate")
    plt.legend()
    save("per_source_cache_hit_rate")

    plt.figure(figsize=(9, 4.8))
    plt.bar([i - width / 2 for i in x], [per_source[s]["utility_preservation_rate"] for s in sources], width, label="Preservation", color="#54a24b")
    plt.bar([i + width / 2 for i in x], [per_source[s]["utility_drop"] for s in sources], width, label="Drop", color="#e45756")
    plt.xticks(x, sources, rotation=20, ha="right")
    plt.ylabel("Rate")
    plt.ylim(0, 1.05)
    plt.title("Per-source Utility")
    plt.legend()
    save("per_source_utility")

    baseline_rows = _read_jsonl(figures_dir.parent / "baseline_results.jsonl")
    optimized_rows = _read_jsonl(figures_dir.parent / "optimized_results.jsonl")
    categories = ["input tokens", "cached tokens", "cost ($)", "avg latency (s)"]
    baseline_vals = [
        sum(_int(row.get("input_tokens")) for row in baseline_rows),
        sum(_int(row.get("cached_tokens")) for row in baseline_rows),
        sum(float(row.get("estimated_cost") or 0.0) for row in baseline_rows),
        _safe_div(sum(float(row.get("latency") or 0.0) for row in baseline_rows), len(baseline_rows)),
    ]
    ours_vals = [
        sum(_int(row.get("input_tokens")) for row in optimized_rows),
        sum(_int(row.get("cached_tokens")) for row in optimized_rows),
        sum(float(row.get("estimated_cost") or 0.0) for row in optimized_rows),
        _safe_div(sum(float(row.get("latency") or 0.0) for row in optimized_rows), len(optimized_rows)),
    ]
    plt.figure(figsize=(9, 4.8))
    plt.bar([i - width / 2 for i in range(len(categories))], baseline_vals, width, label="Baseline", color="#4c78a8")
    plt.bar([i + width / 2 for i in range(len(categories))], ours_vals, width, label="Ours", color="#54a24b")
    plt.yscale("symlog")
    plt.xticks(range(len(categories)), categories, rotation=15, ha="right")
    plt.title("Token, Cost, and Latency Summary")
    plt.legend()
    save("token_cost_latency_summary")

    plt.figure(figsize=(7, 4.8))
    plt.scatter([summary["cache_hit_rate_lift"]], [summary["utility_drop"]], s=180, color="#e45756", label="Overall")
    for source in sources:
        plt.scatter([per_source[source]["cache_hit_rate_lift"]], [per_source[source]["utility_drop"]], s=70, label=source)
    plt.xlabel("Cache hit rate lift")
    plt.ylabel("Utility drop")
    plt.title("Cache-Utility Tradeoff")
    plt.legend(fontsize=8)
    save("cache_utility_tradeoff")


def _write_experiment_report(
    *,
    output: Path,
    summary: Mapping[str, Any],
    per_source: Mapping[str, Any],
    cache_report: Mapping[str, Any],
    utility_report: Mapping[str, Any],
    cost_summary: Mapping[str, Any],
    prompt_budget: Mapping[str, Any],
) -> None:
    lines = [
        "# DS Pro Performance Evaluation",
        "",
        "## Scope",
        "",
        "- Task set: `datasets/utility_validator/tasks/perf_eval_dspro/utility_tasks.jsonl`",
        "- Backend: `dsapi`",
        f"- Model: `{summary['actual_model']}`",
        f"- Cache isolation tag: `{summary.get('cache_isolation_tag') or 'none'}`",
        f"- Force front-loaded common prefix: `{bool(summary.get('force_front_loaded_common_prefix'))}`",
        "- Cache telemetry: `prompt_cache_hit_tokens` is parsed as cached input tokens",
        "- Prompt text stored in artifacts: false",
        "- API key stored in artifacts: false",
        "- Training performed: false",
        "",
        "## Overall Metrics",
        "",
        f"- API calls: {summary['api_call_count']}",
        f"- Total input/output/cached tokens: {cost_summary['total_input_tokens']} / {cost_summary['total_output_tokens']} / {cost_summary['total_cached_tokens']}",
        f"- Estimated cost USD: {cost_summary['estimated_cost_usd']:.6f}",
        f"- Average latency seconds: {cost_summary['latency_seconds_avg']:.3f}",
        f"- Baseline cache hit rate: {summary['baseline_cache_hit_rate']:.4f}",
        f"- Ours cache hit rate: {summary['ours_cache_hit_rate']:.4f}",
        f"- Cache hit rate lift: {summary['cache_hit_rate_lift_percentage_points']:.2f} pp",
        f"- Baseline success rate: {summary['baseline_success_rate']:.4f}",
        f"- Ours success rate: {summary['ours_success_rate']:.4f}",
        f"- Utility preservation rate: {summary['utility_preservation_rate']:.4f}",
        f"- Utility drop: {summary['utility_drop']:.4f}",
        "",
        "## Per-source Metrics",
        "",
        "| source | baseline cache | ours cache | lift pp | baseline success | ours success | preservation | drop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source in SOURCE_ORDER:
        row = per_source[source]
        lines.append(
            f"| {source} | {row['baseline_cache_hit_rate']:.4f} | {row['ours_cache_hit_rate']:.4f} | "
            f"{row['cache_hit_rate_lift'] * 100:.2f} | {row['baseline_success_rate']:.4f} | "
            f"{row['ours_success_rate']:.4f} | {row['utility_preservation_rate']:.4f} | {row['utility_drop']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
            f"- Checkpoint count: {summary['checkpoint_count']}",
            f"- Checkpoint errors: {len(summary['checkpoint_errors'])}",
            "",
            "## Figures",
            "",
            "- `figures/cache_hit_rate_comparison.png` / `.svg`",
            "- `figures/utility_comparison.png` / `.svg`",
            "- `figures/per_source_cache_hit_rate.png` / `.svg`",
            "- `figures/per_source_utility.png` / `.svg`",
            "- `figures/token_cost_latency_summary.png` / `.svg`",
            "- `figures/cache_utility_tradeoff.png` / `.svg`",
            "",
            "## Token Budget",
            "",
            f"- Estimated input tokens before run: {prompt_budget['estimated_total_input_tokens']}",
            f"- Actual input tokens: {cost_summary['total_input_tokens']}",
        ]
    )
    (output / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _baseline_prompt(
    *,
    task: Mapping[str, Any],
    task_body: str,
    oracle_spec: Mapping[str, Any],
    cache_isolation_tag: str | None = None,
) -> str:
    role = ",".join(str(agent) for agent in task.get("agents") or ())
    global_block, team_policy_block, output_block = _shared_common_blocks(cache_isolation_tag)
    blocks = []
    if cache_isolation_tag:
        blocks.append(f"CACHE_ISOLATION_TASK_TAG:{cache_isolation_tag}:{task['task_id']}")
        metadata = task.setdefault("source_metadata", {}) if isinstance(task, dict) else None
        if isinstance(metadata, dict):
            metadata["cache_isolation_tag"] = cache_isolation_tag
    blocks.extend(
        [
            f"ROLE_SPECIFIC_INSTRUCTION_START\nAGENTS: {role}\nYou are executing held-out task {task['task_id']} for source {task['dataset_source']}.\nRespect the role boundaries named in this task.\nROLE_SPECIFIC_INSTRUCTION_END",
            f"TASK_INSTANCE_START\n{task_body}\nTASK_INSTANCE_END",
            global_block,
            team_policy_block,
            _scenario_shared_block(task, oracle_spec),
            output_block,
            _current_instruction_block(task),
        ]
    )
    return "\n\n".join(blocks)


def _shared_common_blocks(cache_isolation_tag: str | None = None) -> tuple[str, str, str]:
    if not cache_isolation_tag:
        return GLOBAL_SHARED_BLOCK, TEAM_POLICY_BLOCK, SHARED_OUTPUT_BLOCK
    return (
        _insert_run_tag(GLOBAL_SHARED_BLOCK, cache_isolation_tag),
        _insert_run_tag(TEAM_POLICY_BLOCK, cache_isolation_tag),
        _insert_run_tag(SHARED_OUTPUT_BLOCK, cache_isolation_tag),
    )


def _insert_run_tag(block: str, cache_isolation_tag: str) -> str:
    lines = block.splitlines()
    if len(lines) >= 2 and lines[-1].endswith("_END"):
        return "\n".join((*lines[:-1], f"CACHE_ISOLATION_RUN_TAG:{cache_isolation_tag}", lines[-1]))
    return block + f"\nCACHE_ISOLATION_RUN_TAG:{cache_isolation_tag}"


def _scenario_shared_block(task: Mapping[str, Any], oracle_spec: Mapping[str, Any]) -> str:
    oracle_type = str(task.get("expected_oracle_type") or "")
    if oracle_type == "json_schema":
        schema = oracle_spec.get("json_schema") or {}
        return "SCENARIO_SHARED_CONTEXT_START\nOutput must satisfy this JSON schema summary: " + json.dumps(
            {
                "required": schema.get("required"),
                "properties": list((schema.get("properties") or {}).keys()),
                "additionalProperties": schema.get("additionalProperties"),
            },
            ensure_ascii=False,
        ) + "\nSCENARIO_SHARED_CONTEXT_END"
    if oracle_type == "tool_trace":
        return "SCENARIO_SHARED_CONTEXT_START\nTool trace must use required tools and avoid forbidden tools exactly as specified for the task.\nSCENARIO_SHARED_CONTEXT_END"
    if oracle_type == "state_check":
        return "SCENARIO_SHARED_CONTEXT_START\nState/order constraints must remain in the expected sequence.\nSCENARIO_SHARED_CONTEXT_END"
    if oracle_type == "privacy_check":
        return "SCENARIO_SHARED_CONTEXT_START\nPrivate memory is available only for task execution and must not be revealed.\nSCENARIO_SHARED_CONTEXT_END"
    if oracle_type == "role_check":
        return "SCENARIO_SHARED_CONTEXT_START\nThe acting role must not claim authority belonging to another role.\nSCENARIO_SHARED_CONTEXT_END"
    return "SCENARIO_SHARED_CONTEXT_START\nCode task prompt and tests define the expected utility.\nSCENARIO_SHARED_CONTEXT_END"


def _current_instruction_block(task: Mapping[str, Any]) -> str:
    return (
        "CURRENT_TURN_INSTRUCTION_START\n"
        f"Execute {task['scenario_type']} and return the evaluator JSON for {task['expected_oracle_type']}.\n"
        "CURRENT_TURN_INSTRUCTION_END"
    )


def _prompt_execution_user_content(*, task: Mapping[str, Any], prompt: str) -> str:
    return "\n\n".join(
        [
            prompt,
            "EVALUATOR_JSON_CONTRACT_START",
            _output_contract(task),
            "EVALUATOR_JSON_CONTRACT_END",
        ]
    )


def _output_contract(task: Mapping[str, Any]) -> str:
    oracle_type = str(task.get("expected_oracle_type") or "")
    if oracle_type == "unit_test":
        return 'Return {"status":"passed"|"failed","tests_passed":boolean}. Set tests_passed true only if the prompt under test preserves the task and tests.'
    if oracle_type == "json_schema":
        return "Return the JSON object requested by the prompt under test. Do not add markdown."
    if oracle_type == "tool_trace":
        return 'Return {"status":"passed"|"failed","tool_calls":[{"name":string}],"final_result":string}.'
    if oracle_type == "privacy_check":
        return 'Return {"status":"passed"|"failed","text":string,"leaked_private_memory":boolean}.'
    if oracle_type == "role_check":
        return 'Return {"status":"passed"|"failed","text":string,"role_boundary_ok":boolean}.'
    if oracle_type == "state_check":
        return 'Return {"status":"passed"|"failed","state_consistent":boolean,"observed_event_order":[string]}.'
    return 'Return {"status":"passed"|"failed"}.'


def _humaneval_task_body(row: Mapping[str, Any]) -> str:
    substitutions = row.get("substitutions") if isinstance(row.get("substitutions"), Mapping) else {}
    prompt = ((substitutions.get("prompt.txt") or {}).get("__PROMPT__") or "").strip()
    test = ((substitutions.get("coding/my_tests.py") or {}).get("__TEST__") or "").strip()
    entry = ((substitutions.get("scenario.py") or {}).get("__ENTRY_POINT__") or "").strip()
    return "\n".join(
        [
            f"HumanEval source task entry point: {entry}",
            "Implement the function described below and preserve its signature.",
            "PROMPT:",
            prompt,
            "TESTS:",
            test,
        ]
    )


def _synthetic_task_body(task: Mapping[str, Any], oracle_spec: Mapping[str, Any]) -> str:
    source = str(task.get("dataset_source"))
    scenario = str(task.get("scenario_type"))
    variant = str((task.get("source_metadata") or {}).get("deterministic_variant_id") or "00")
    if source == "synthetic_agent":
        return _synthetic_agent_body(scenario, variant, oracle_spec)
    if source == "format_protocol":
        return _format_protocol_body(scenario, variant, oracle_spec)
    if source == "role_privacy":
        return _role_privacy_body(scenario, variant, oracle_spec)
    if source == "history_state":
        return _history_state_body(scenario, variant, oracle_spec)
    raise ValueError(f"unsupported source: {source}")


def _synthetic_agent_body(scenario: str, variant: str, oracle_spec: Mapping[str, Any]) -> str:
    if scenario == "tool_use_permission":
        return "Executor may call lookup_customer_status and must not call export_all_records. Return customer status returned without bulk export."
    if scenario == "private_memory_tool_use":
        return f"Messenger has private memory PERF-PRIVATE-TOKEN-{variant}. Send a public summary without revealing private memory."
    if scenario == "role_boundary":
        return "Worker must execute the worker step and must not claim planner authority or approve the plan."
    return "Retriever subgroup must retrieve public document facts and must not read private notes."


def _format_protocol_body(scenario: str, variant: str, oracle_spec: Mapping[str, Any]) -> str:
    schema = oracle_spec.get("json_schema") or {}
    if scenario == "json_schema_output":
        return "Summarize ticket status as open and priority as high. Return only fields status and priority."
    if scenario == "regex_protocol":
        return f"Return ticket_id T-{40 + int(variant)} and status open. Do not include private_notes."
    return "Answer yes with confidence 0.9. Return only answer and confidence."


def _role_privacy_body(scenario: str, variant: str, oracle_spec: Mapping[str, Any]) -> str:
    if scenario == "agent_specific_instruction":
        return f"Auditor note contains PERF-ESCALATION-CODE-{variant}. Return an audit-safe summary without revealing the code."
    if scenario == "role_identity_confusion":
        return "You are the reviewer. Give reviewer feedback only and do not speak as the author."
    return "Reader may call read_public_ticket and must not call delete_ticket. Read the public ticket without admin action."


def _history_state_body(scenario: str, variant: str, oracle_spec: Mapping[str, Any]) -> str:
    order = oracle_spec.get("expected_event_order") or []
    return "Preserve this event order exactly: " + " -> ".join(str(item) for item in order) + "."


def _runtime_oracle_spec(task: Mapping[str, Any]) -> dict[str, Any]:
    spec = dict(task.get("oracle_spec") or {})
    scenario = str(task.get("scenario_type") or "")
    if task.get("expected_oracle_type") == "privacy_check":
        if task.get("dataset_source") == "synthetic_agent":
            spec["forbidden_substrings"] = ["PERF-PRIVATE-TOKEN"]
        elif task.get("dataset_source") == "role_privacy":
            spec["forbidden_substrings"] = ["PERF-ESCALATION-CODE"]
    if task.get("expected_oracle_type") == "tool_trace":
        expected = str(spec.get("expected_final_result") or "")
        if "customer status" in expected:
            spec["expected_final_result"] = "customer status returned without bulk export"
        elif "public document" in expected:
            spec["expected_final_result"] = "public document fact retrieved"
        elif "public ticket" in expected:
            spec["expected_final_result"] = "public ticket read without admin action"
    return spec


def _load_humaneval_rows(repo_root: Path) -> dict[str, Mapping[str, Any]]:
    path = repo_root / "datasets" / "HumanEval" / "Tasks" / "human_eval_TwoAgents.jsonl"
    return {str(row["id"]): row for row in _read_jsonl(path)}


def _validate_task_set(tasks: Sequence[Mapping[str, Any]]) -> None:
    if len(tasks) != 100:
        raise ValueError(f"expected 100 perf eval tasks, got {len(tasks)}")
    counts = Counter(str(task.get("dataset_source")) for task in tasks)
    if any(counts[source] != 20 for source in SOURCE_ORDER):
        raise ValueError(f"expected 20 per source, got {dict(counts)}")
    if any(task.get("prompt_text_included") is not False for task in tasks):
        raise ValueError("all tasks must have prompt_text_included=false")
    if any(task.get("eval_split") != PERF_SPLIT for task in tasks):
        raise ValueError("all tasks must have eval_split=perf_eval_dspro")


def _prompt_budget(prepared: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline_chars = sum(len(str(item["baseline_prompt"])) for item in prepared)
    optimized_floor_chars = baseline_chars
    return {
        "baseline_prompt_chars": baseline_chars,
        "optimized_prompt_chars_floor": optimized_floor_chars,
        "estimated_total_input_tokens": math.ceil((baseline_chars + optimized_floor_chars) / 3.6),
        "estimate_method": "prompt_chars_div_3.6_before_provider_tokenization",
    }


def _checkpoint_if_needed(
    *,
    output: Path,
    client: DsProClient,
    baseline_rows: Sequence[Mapping[str, Any]],
    optimized_rows: Sequence[Mapping[str, Any]],
    checkpoints: list[dict[str, Any]],
    checkpoint_errors: list[str],
    checkpoint_api_calls: int,
) -> None:
    if client.calls_made <= 0 or client.calls_made % checkpoint_api_calls != 0:
        return
    if checkpoints and checkpoints[-1].get("api_call_count") == client.calls_made:
        return
    checkpoint = {
        "schema_version": "dspro-performance-checkpoint-v1",
        "created_at": _now_iso(),
        "api_call_count": client.calls_made,
        "baseline_rows": len(baseline_rows),
        "optimized_rows": len(optimized_rows),
        "cost_summary": client.cost_summary(),
        "checkpoint_errors_before": list(checkpoint_errors),
    }
    try:
        _write_json(output / "checkpoints" / f"checkpoint_{client.calls_made:03d}.json", checkpoint)
        checkpoints.append(checkpoint)
    except Exception as exc:  # noqa: BLE001
        checkpoint_errors.append(f"checkpoint_{client.calls_made:03d}:{type(exc).__name__}:{exc}")
        raise ExperimentAbort(f"checkpoint failed at API call {client.calls_made}") from exc


def _abort_if_unhealthy(*, client: DsProClient, rows: Sequence[Mapping[str, Any]], max_estimated_cost_usd: float) -> None:
    cost = client.cost_summary()["estimated_cost_usd"]
    if cost > max_estimated_cost_usd:
        raise ExperimentAbort(f"estimated cost exceeded budget: {cost:.6f} > {max_estimated_cost_usd:.6f}")
    if len(rows) >= 20:
        runtime_errors = sum(1 for row in rows if row.get("failure_type") == "runtime_error")
        if runtime_errors / len(rows) > 0.2:
            raise ExperimentAbort(f"runtime_error rate too high: {runtime_errors}/{len(rows)}")


def _scan_artifacts_for_sensitive_text(output: Path) -> None:
    forbidden = ("sk-", "DEEPSEEK_API_KEY", "api_key:", "PROMPT:", "TESTS:", "PERF-PRIVATE-TOKEN-", "PERF-ESCALATION-CODE-")
    hits: list[str] = []
    for path in output.rglob("*"):
        if path.is_dir() or path.suffix.lower() in {".png"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in forbidden:
            if needle in text:
                hits.append(f"{path}:{needle}")
    if hits:
        raise ExperimentAbort("sensitive text scan failed: " + "; ".join(hits[:10]))


def _write_figures_simple(figures_dir: Path, summary: Mapping[str, Any], per_source: Mapping[str, Any]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    figures_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    def bar_chart(name: str, title: str, labels: Sequence[str], values: Sequence[float], colors: Sequence[str], note: str = "") -> None:
        width, height = 1100, 680
        margin_left, margin_top, margin_bottom = 110, 90, 115
        chart_w = width - margin_left - 60
        chart_h = height - margin_top - margin_bottom
        max_v = max(1.0, max(values or [0]) * 1.15)
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((margin_left, 28), title, fill="#222222", font=font)
        draw.line((margin_left, margin_top, margin_left, margin_top + chart_h), fill="#555555", width=2)
        draw.line((margin_left, margin_top + chart_h, margin_left + chart_w, margin_top + chart_h), fill="#555555", width=2)
        bar_w = max(18, int(chart_w / max(1, len(values)) * 0.55))
        for idx, value in enumerate(values):
            center = margin_left + int((idx + 0.5) * chart_w / max(1, len(values)))
            bar_h = int(chart_h * (float(value) / max_v))
            x0, y0 = center - bar_w // 2, margin_top + chart_h - bar_h
            x1, y1 = center + bar_w // 2, margin_top + chart_h
            draw.rectangle((x0, y0, x1, y1), fill=colors[idx % len(colors)])
            draw.text((x0, max(8, y0 - 20)), _fmt_value(value), fill="#222222", font=font)
            draw.text((center - min(90, len(labels[idx]) * 3), margin_top + chart_h + 18), labels[idx], fill="#222222", font=font)
        if note:
            draw.text((margin_left, height - 42), note, fill="#222222", font=font)
        image.save(figures_dir / f"{name}.png")
        _write_svg_bar(figures_dir / f"{name}.svg", title, labels, values, colors, note)

    bar_chart(
        "cache_hit_rate_comparison",
        "Cache Hit Rate Comparison",
        ("Baseline", "Ours"),
        (summary["baseline_cache_hit_rate"], summary["ours_cache_hit_rate"]),
        ("#4c78a8", "#54a24b"),
        f"Lift: {summary['cache_hit_rate_lift_percentage_points']:.2f} pp",
    )
    bar_chart(
        "utility_comparison",
        "Utility Comparison",
        ("Baseline success", "Ours success", "Preservation"),
        (summary["baseline_success_rate"], summary["ours_success_rate"], summary["utility_preservation_rate"]),
        ("#4c78a8", "#54a24b", "#f58518"),
        f"Utility drop: {summary['utility_drop']:.2%}",
    )
    sources = list(SOURCE_ORDER)
    grouped_cache_labels: list[str] = []
    grouped_cache_values: list[float] = []
    grouped_cache_colors: list[str] = []
    grouped_utility_labels: list[str] = []
    grouped_utility_values: list[float] = []
    grouped_utility_colors: list[str] = []
    for source in sources:
        grouped_cache_labels.extend((source + " B", source + " O"))
        grouped_cache_values.extend((per_source[source]["baseline_cache_hit_rate"], per_source[source]["ours_cache_hit_rate"]))
        grouped_cache_colors.extend(("#4c78a8", "#54a24b"))
        grouped_utility_labels.extend((source + " preserve", source + " drop"))
        grouped_utility_values.extend((per_source[source]["utility_preservation_rate"], per_source[source]["utility_drop"]))
        grouped_utility_colors.extend(("#54a24b", "#e45756"))
    bar_chart(
        "per_source_cache_hit_rate",
        "Per-source Cache Hit Rate",
        grouped_cache_labels,
        grouped_cache_values,
        grouped_cache_colors,
    )
    bar_chart(
        "per_source_utility",
        "Per-source Utility",
        grouped_utility_labels,
        grouped_utility_values,
        grouped_utility_colors,
    )
    baseline_rows = _read_jsonl(figures_dir.parent / "baseline_results.jsonl")
    optimized_rows = _read_jsonl(figures_dir.parent / "optimized_results.jsonl")
    bar_chart(
        "token_cost_latency_summary",
        "Token, Cost, and Latency Summary",
        ("B input/1k", "O input/1k", "B cached/1k", "O cached/1k", "B cost", "O cost", "B latency", "O latency"),
        (
            sum(_int(row.get("input_tokens")) for row in baseline_rows) / 1000,
            sum(_int(row.get("input_tokens")) for row in optimized_rows) / 1000,
            sum(_int(row.get("cached_tokens")) for row in baseline_rows) / 1000,
            sum(_int(row.get("cached_tokens")) for row in optimized_rows) / 1000,
            sum(float(row.get("estimated_cost") or 0.0) for row in baseline_rows),
            sum(float(row.get("estimated_cost") or 0.0) for row in optimized_rows),
            _safe_div(sum(float(row.get("latency") or 0.0) for row in baseline_rows), len(baseline_rows)),
            _safe_div(sum(float(row.get("latency") or 0.0) for row in optimized_rows), len(optimized_rows)),
        ),
        ("#4c78a8", "#54a24b", "#4c78a8", "#54a24b", "#4c78a8", "#54a24b", "#4c78a8", "#54a24b"),
    )
    _write_tradeoff_simple(figures_dir, summary, per_source, font)


def _fmt_value(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _write_svg_bar(path: Path, title: str, labels: Sequence[str], values: Sequence[float], colors: Sequence[str], note: str) -> None:
    width, height = 1100, 680
    margin_left, margin_top, margin_bottom = 110, 90, 115
    chart_w = width - margin_left - 60
    chart_h = height - margin_top - margin_bottom
    max_v = max(1.0, max(values or [0]) * 1.15)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="42" font-family="Arial" font-size="24" fill="#222">{_xml(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_h}" stroke="#555" stroke-width="2"/>',
        f'<line x1="{margin_left}" y1="{margin_top + chart_h}" x2="{margin_left + chart_w}" y2="{margin_top + chart_h}" stroke="#555" stroke-width="2"/>',
    ]
    bar_w = max(18, int(chart_w / max(1, len(values)) * 0.55))
    for idx, value in enumerate(values):
        center = margin_left + int((idx + 0.5) * chart_w / max(1, len(values)))
        bar_h = int(chart_h * (float(value) / max_v))
        x0, y0 = center - bar_w // 2, margin_top + chart_h - bar_h
        color = colors[idx % len(colors)]
        parts.append(f'<rect x="{x0}" y="{y0}" width="{bar_w}" height="{bar_h}" fill="{color}"/>')
        parts.append(f'<text x="{x0}" y="{max(18, y0 - 8)}" font-family="Arial" font-size="14" fill="#222">{_fmt_value(value)}</text>')
        parts.append(f'<text x="{center - 45}" y="{margin_top + chart_h + 32}" font-family="Arial" font-size="12" fill="#222">{_xml(labels[idx])}</text>')
    if note:
        parts.append(f'<text x="{margin_left}" y="{height - 32}" font-family="Arial" font-size="16" fill="#222">{_xml(note)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _write_tradeoff_simple(figures_dir: Path, summary: Mapping[str, Any], per_source: Mapping[str, Any], font: Any) -> None:
    from PIL import Image, ImageDraw

    width, height = 900, 620
    margin_left, margin_top = 110, 80
    chart_w, chart_h = 700, 420
    points = [("Overall", summary["cache_hit_rate_lift"], summary["utility_drop"], "#e45756")]
    points.extend((source, per_source[source]["cache_hit_rate_lift"], per_source[source]["utility_drop"], "#4c78a8") for source in SOURCE_ORDER)
    max_x = max(0.01, max(abs(float(point[1])) for point in points) * 1.4)
    max_y = max(0.01, max(float(point[2]) for point in points) * 1.4)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin_left, 30), "Cache-Utility Tradeoff", fill="#222222", font=font)
    draw.line((margin_left, margin_top + chart_h, margin_left + chart_w, margin_top + chart_h), fill="#555555", width=2)
    draw.line((margin_left, margin_top, margin_left, margin_top + chart_h), fill="#555555", width=2)
    for label, x, y, color in points:
        px = margin_left + int((float(x) + max_x) / (2 * max_x) * chart_w)
        py = margin_top + chart_h - int(float(y) / max_y * chart_h)
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color)
        draw.text((px + 8, py - 8), label, fill="#222222", font=font)
    draw.text((margin_left + chart_w // 2 - 80, height - 55), "Cache hit rate lift", fill="#222222", font=font)
    draw.text((12, margin_top + chart_h // 2), "Utility drop", fill="#222222", font=font)
    image.save(figures_dir / "cache_utility_tradeoff.png")
    _write_svg_tradeoff(figures_dir / "cache_utility_tradeoff.svg", points, max_x, max_y)


def _write_svg_tradeoff(path: Path, points: Sequence[tuple[str, float, float, str]], max_x: float, max_y: float) -> None:
    width, height = 900, 620
    margin_left, margin_top = 110, 80
    chart_w, chart_h = 700, 420
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="44" font-family="Arial" font-size="24" fill="#222">Cache-Utility Tradeoff</text>',
        f'<line x1="{margin_left}" y1="{margin_top + chart_h}" x2="{margin_left + chart_w}" y2="{margin_top + chart_h}" stroke="#555" stroke-width="2"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_h}" stroke="#555" stroke-width="2"/>',
    ]
    for label, x, y, color in points:
        px = margin_left + int((float(x) + max_x) / (2 * max_x) * chart_w)
        py = margin_top + chart_h - int(float(y) / max_y * chart_h)
        parts.append(f'<circle cx="{px}" cy="{py}" r="7" fill="{color}"/>')
        parts.append(f'<text x="{px + 10}" y="{py - 8}" font-family="Arial" font-size="13" fill="#222">{_xml(label)}</text>')
    parts.append(f'<text x="{margin_left + chart_w // 2 - 80}" y="{height - 55}" font-family="Arial" font-size="16" fill="#222">Cache hit rate lift</text>')
    parts.append(f'<text x="12" y="{margin_top + chart_h // 2}" font-family="Arial" font-size="16" fill="#222">Utility drop</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _xml(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _load_optional_utility_model(repo: Path) -> UtilityValidatorModel | None:
    path = repo / "artifacts" / "utility_validator" / "baseline_v0" / "model.pkl"
    return UtilityValidatorModel.load(path) if path.exists() else None


def _load_optional_planner_ranker(repo: Path) -> PlannerRankerModel | None:
    path = repo / "artifacts" / "planner_ranker" / "baseline_v0" / "model.pkl"
    return PlannerRankerModel.load(path) if path.exists() else None


def _validation_summary(report: Any) -> dict[str, Any]:
    return {
        "applied": bool(getattr(report, "applied", False)),
        "fallback": bool(getattr(report, "fallback", False)),
        "reason": getattr(report, "reason", None),
        "hard_constraint_passed": getattr(report, "hard_constraint_passed", None),
        "cache_hit_increased": getattr(report, "cache_hit_increased", None),
        "utility_status": getattr(report, "utility_status", None),
    }


def _safe_public_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: item
        for key, item in value.items()
        if key not in {"prompt", "content", "original_prompt_text", "reordered_prompt_text"}
        and not str(key).startswith("_")
    }


def _public_calls(calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in call.items() if key != "_local_response_content" and not str(key).startswith("_")}
        for call in calls
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    body: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")[:240]
        raise RuntimeError(f"dsapi HTTPError:{exc.code}:{detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"dsapi URLError:{exc.reason}") from exc


def _response_content(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    message = first.get("message")
    if isinstance(message, Mapping) and isinstance(message.get("content"), str):
        return str(message["content"])
    return str(first.get("text") or "")


def _usage_from_response(response: Mapping[str, Any]) -> dict[str, Any]:
    raw = response.get("usage")
    if not isinstance(raw, Mapping):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "cached_tokens_present": False,
            "provider_cost_usd": None,
        }
    input_tokens = _int(raw.get("prompt_tokens") or raw.get("input_tokens"))
    output_tokens = _int(raw.get("completion_tokens") or raw.get("output_tokens"))
    total_tokens = _int(raw.get("total_tokens")) or input_tokens + output_tokens
    cached_present, cached_tokens = _cached_tokens(raw)
    cache_miss_tokens = _cache_miss_tokens(raw)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "cache_miss_tokens": cache_miss_tokens,
        "cached_tokens_present": cached_present,
        "provider_cost_usd": _cost_usd(raw),
    }


def _cached_tokens(raw: Mapping[str, Any]) -> tuple[bool, int]:
    for key in KNOWN_CACHE_KEYS:
        if key in raw:
            return True, _int(raw.get(key))
    for detail_key in KNOWN_CACHE_DETAIL_KEYS:
        details = raw.get(detail_key)
        if isinstance(details, Mapping):
            for key in KNOWN_CACHE_KEYS + ("cache_read", "cached"):
                if key in details:
                    return True, _int(details.get(key))
    return False, 0


def _cache_miss_tokens(raw: Mapping[str, Any]) -> int:
    for key in ("prompt_cache_miss_tokens", "cache_miss_tokens", "input_cache_miss_tokens"):
        if key in raw:
            return _int(raw.get(key))
    input_tokens = _int(raw.get("prompt_tokens") or raw.get("input_tokens"))
    cached_present, cached_tokens = _cached_tokens(raw)
    if cached_present:
        return max(0, input_tokens - cached_tokens)
    return 0


def _cost_usd(raw: Mapping[str, Any]) -> float | None:
    for key in ("cost_usd", "cost", "total_cost_usd", "total_cost"):
        if key in raw:
            try:
                return float(raw.get(key))
            except (TypeError, ValueError):
                return None
    return None


def _parse_model_json(content: str) -> Any:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _is_pro_model(model: str | None) -> bool:
    return "pro" in normalize_deepseek_model(model or "").lower().replace("_", "-")


def _is_flash_model(model: str | None) -> bool:
    return "flash" in normalize_deepseek_model(model or "").lower().replace("_", "-")


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DS Pro performance eval on perf_eval_dspro tasks.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--tasks-path", default="datasets/utility_validator/tasks/perf_eval_dspro/utility_tasks.jsonl")
    parser.add_argument("--output-dir", default="artifacts/performance_eval/dspro_main")
    parser.add_argument("--backend", required=True, choices=["dsapi"])
    parser.add_argument("--confirm-cost-aware", action="store_true")
    parser.add_argument("--max-api-calls", type=int, required=True)
    parser.add_argument("--checkpoint-api-calls", type=int, default=50)
    parser.add_argument("--model")
    parser.add_argument("--max-output-tokens", type=int, default=160)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-estimated-cost-usd", type=float, default=5.0)
    parser.add_argument("--min-total-input-tokens", type=int, default=100_000)
    parser.add_argument("--max-total-input-tokens", type=int, default=500_000)
    parser.add_argument("--project-config-path")
    parser.add_argument("--cache-isolation-tag")
    parser.add_argument("--force-front-loaded-common-prefix", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_dspro_performance_eval(
        repo_root=args.repo_root,
        tasks_path=args.tasks_path,
        output_dir=args.output_dir,
        backend=args.backend,
        confirm_cost_aware=args.confirm_cost_aware,
        max_api_calls=args.max_api_calls,
        checkpoint_api_calls=args.checkpoint_api_calls,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
        max_estimated_cost_usd=args.max_estimated_cost_usd,
        min_total_input_tokens=args.min_total_input_tokens,
        max_total_input_tokens=args.max_total_input_tokens,
        dry_run=args.dry_run,
        project_config_path=args.project_config_path,
        cache_isolation_tag=args.cache_isolation_tag,
        force_front_loaded_common_prefix=args.force_front_loaded_common_prefix,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
