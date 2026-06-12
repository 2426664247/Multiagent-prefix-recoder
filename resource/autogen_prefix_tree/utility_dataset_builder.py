from __future__ import annotations

import argparse
import copy
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from autogen_core.models import SystemMessage, UserMessage

from .compiler import LocalPromptCompiler
from .ir import stable_hash
from .planner import HierarchicalPrefixPlanner, rewrite_messages
from .project_api_config import load_project_provider_config, normalize_deepseek_model
from .telemetry import dataclass_to_dict, serialize_prefix_tree_candidate
from .utility_dataset_sources import DATASET_SOURCES, build_utility_smoke_tasks
from .utility_oracles import OracleResult, oracle_for_task
from .validator import CacheUtilityValidator


@dataclass(frozen=True)
class UtilityDatasetBuildResult:
    output_root: str
    task_path: str
    label_path: str
    feature_path: str
    summary_path: str
    summary: Mapping[str, Any]


REAL_DATASET_MODES = {"smoke_real", "pilot_real"}


class FakeModelBackend:
    name = "fake"
    label_source = "fake_smoke_oracle"
    model_name = None

    def generate(
        self,
        *,
        task: Mapping[str, Any],
        prompt: str,
        phase: str,
        candidate: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        oracle_type = str(task.get("expected_oracle_type") or "")
        task_id = str(task.get("task_id") or "")
        if oracle_type == "unit_test":
            return {"status": "passed", "tests_passed": True, "task_id": task_id}, {"tool_calls": ("run_tests",)}
        if oracle_type == "json_schema":
            return _fake_json_schema_output(task), None
        if oracle_type == "tool_trace":
            spec = task.get("oracle_spec") if isinstance(task.get("oracle_spec"), Mapping) else {}
            required = tuple(spec.get("required_tool_calls") or ())
            final_result = spec.get("expected_final_result") or "ok"
            return {
                "status": "passed",
                "tool_calls": tuple({"name": name} for name in required),
                "final_result": final_result,
            }, {"tool_calls": tuple({"name": name} for name in required)}
        if oracle_type == "privacy_check":
            return {
                "status": "passed",
                "text": "Public summary only. No private memory is included.",
                "leaked_private_memory": False,
            }, None
        if oracle_type == "role_check":
            return {
                "status": "passed",
                "text": "Reviewer-safe or role-safe result.",
                "role_boundary_ok": True,
            }, None
        if oracle_type == "state_check":
            spec = task.get("oracle_spec") if isinstance(task.get("oracle_spec"), Mapping) else {}
            order = tuple(spec.get("expected_event_order") or ())
            return {"status": "passed", "state_consistent": True, "observed_event_order": order}, {"events": order}
        return {"status": "passed"}, None

    def cost_summary(self) -> Mapping[str, Any]:
        return {
            "api_call_count": 0,
            "max_api_calls": None,
            "model": None,
            "model_is_v4_flash": False,
            "model_is_pro": False,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cached_tokens": 0,
            "cached_tokens": 0,
            "estimated_cost_usd": 0.0,
            "estimated_cost": 0.0,
            "latency_seconds_total": 0.0,
            "latency_seconds_avg": 0.0,
            "calls": (),
        }

    def remaining_calls(self) -> int | None:
        return None


class DsApiBackend:
    name = "dsapi"
    label_source = "dsapi_execution_oracle"

    def __init__(
        self,
        *,
        repo_root: str | Path,
        max_api_calls: int | None,
        confirm_cost_aware: bool,
        max_output_tokens: int = 96,
        timeout_seconds: float = 120.0,
        price_input_per_million: float | None = None,
        price_output_per_million: float | None = None,
        config_path: str | Path | None = None,
        model_override: str | None = None,
    ) -> None:
        if not confirm_cost_aware or max_api_calls is None or max_api_calls <= 0:
            raise ValueError(
                "dsapi backend requires --backend dsapi --max-api-calls N --confirm-cost-aware; "
                "no API call was made"
            )
        config = load_project_provider_config(cwd=repo_root, config_path=config_path)
        if config is None:
            raise ValueError("dsapi backend requires DEEPSEEK_API_KEY or config/config.txt; no API call was made")
        effective_model = _normalize_dsapi_model(model_override or config.model)
        if _is_pro_model(effective_model) or not _is_v4_flash_model(effective_model):
            raise ValueError(
                "dsapi backend requires DeepSeek V4 Flash/V4Flash; "
                f"effective model is {effective_model!r}; no API call was made"
            )
        self.max_api_calls = max_api_calls
        self.confirm_cost_aware = confirm_cost_aware
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.config = config
        self.model_name = effective_model
        self.model_override_used = bool(model_override)
        self.price_input_per_million = (
            _default_input_price_per_million(effective_model)
            if price_input_per_million is None
            else float(price_input_per_million)
        )
        self.price_output_per_million = (
            _default_output_price_per_million(effective_model)
            if price_output_per_million is None
            else float(price_output_per_million)
        )
        self.calls_made = 0
        self.call_reports: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        task: Mapping[str, Any],
        prompt: str,
        phase: str,
        candidate: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        if self.calls_made >= self.max_api_calls:
            raise RuntimeError(f"dsapi max api calls exceeded: {self.max_api_calls}")
        body = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": _dsapi_system_instruction(task),
                },
                {
                    "role": "user",
                    "content": _dsapi_user_content(task, prompt),
                },
            ],
            "temperature": 0,
            "stream": False,
            "max_tokens": self.max_output_tokens,
        }
        if str(task.get("expected_oracle_type") or "") in {
            "unit_test",
            "json_schema",
            "tool_trace",
            "privacy_check",
            "role_check",
            "state_check",
        }:
            body["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        response = _post_chat_completion(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            body=body,
            timeout_seconds=self.timeout_seconds,
        )
        latency = time.perf_counter() - started
        self.calls_made += 1
        usage = _usage_from_response(response)
        estimated_cost = self._estimated_cost_usd(usage)
        content = _response_content(response)
        call_report = {
            "call_index": self.calls_made,
            "task_id": task.get("task_id"),
            "phase": phase,
            "model": self.model_name,
            "base_url": self.config.base_url,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "cached_tokens": usage["cached_tokens"],
            "latency_seconds": latency,
            "latency": latency,
            "estimated_cost_usd": estimated_cost,
            "estimated_cost": estimated_cost,
            "provider_cost_usd": usage["provider_cost_usd"],
            "status": "ok",
        }
        self.call_reports.append(call_report)
        trace = {
            "dsapi_call": {
                "call_index": self.calls_made,
                "task_id": task.get("task_id"),
                "phase": phase,
                "model": self.model_name,
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "total_tokens": usage["total_tokens"],
                "cached_tokens": usage["cached_tokens"],
                "latency_seconds": latency,
                "latency": latency,
                "estimated_cost_usd": estimated_cost,
                "estimated_cost": estimated_cost,
                "provider_cost_usd": usage["provider_cost_usd"],
            }
        }
        return content, trace

    def cost_summary(self) -> Mapping[str, Any]:
        total_latency = sum(float(call.get("latency_seconds") or 0.0) for call in self.call_reports)
        total_cached_tokens = sum(_int(call.get("cached_tokens")) for call in self.call_reports)
        estimated_cost = sum(float(call.get("estimated_cost_usd") or 0.0) for call in self.call_reports)
        return {
            "api_call_count": self.calls_made,
            "max_api_calls": self.max_api_calls,
            "model": self.model_name,
            "model_override_used": self.model_override_used,
            "model_is_v4_flash": _is_v4_flash_model(self.model_name),
            "model_is_pro": _is_pro_model(self.model_name),
            "provider": self.config.provider,
            "base_url": self.config.base_url,
            "config_source": "env" if self.config.config_path is None else "config/config.txt",
            "api_key_stored": False,
            "total_input_tokens": sum(_int(call.get("input_tokens")) for call in self.call_reports),
            "total_output_tokens": sum(_int(call.get("output_tokens")) for call in self.call_reports),
            "total_cached_tokens": total_cached_tokens,
            "cached_tokens": total_cached_tokens,
            "total_tokens": sum(_int(call.get("total_tokens")) for call in self.call_reports),
            "estimated_cost_usd": estimated_cost,
            "estimated_cost": estimated_cost,
            "provider_cost_usd": sum(float(call.get("provider_cost_usd") or 0.0) for call in self.call_reports),
            "price_input_per_million": self.price_input_per_million,
            "price_output_per_million": self.price_output_per_million,
            "latency_seconds_total": total_latency,
            "latency_seconds_avg": total_latency / self.calls_made if self.calls_made else 0.0,
            "calls": tuple(self.call_reports),
        }

    def remaining_calls(self) -> int:
        return self.max_api_calls - self.calls_made

    def _estimated_cost_usd(self, usage: Mapping[str, Any]) -> float:
        provider_cost = usage.get("provider_cost_usd")
        if provider_cost is not None:
            return float(provider_cost)
        input_tokens = max(0, _int(usage.get("input_tokens")) - _int(usage.get("cached_tokens")))
        output_tokens = _int(usage.get("output_tokens"))
        return (
            input_tokens / 1_000_000 * self.price_input_per_million
            + output_tokens / 1_000_000 * self.price_output_per_million
        )


def build_utility_validator_smoke_dataset(
    *,
    repo_root: str | Path = ".",
    output_root: str | Path = "datasets/utility_validator",
    source: str = "all",
    mode: str = "smoke",
    backend: str = "fake",
    max_tasks: int = 15,
    include_text: bool = False,
    max_api_calls: int | None = None,
    confirm_cost_aware: bool = False,
    max_output_tokens: int = 96,
    timeout_seconds: float = 120.0,
    price_input_per_million: float | None = None,
    price_output_per_million: float | None = None,
    project_config_path: str | Path | None = None,
    model: str | None = None,
    created_at: str | None = None,
) -> UtilityDatasetBuildResult:
    if mode not in {"smoke", "smoke_real", "pilot_real"}:
        raise ValueError("only smoke, smoke_real, and pilot_real modes are implemented in the dataset builder")
    if backend == "fake" and mode in REAL_DATASET_MODES:
        raise ValueError(f"{mode} requires --backend dsapi")
    if backend == "fake":
        backend_impl: Any = FakeModelBackend()
        effective_mode = mode
    elif backend == "dsapi":
        if not confirm_cost_aware or max_api_calls is None or max_api_calls <= 0:
            raise ValueError(
                "dsapi backend requires --backend dsapi --max-api-calls N --confirm-cost-aware; "
                "no API call was made"
            )
        if mode not in REAL_DATASET_MODES:
            raise ValueError("dsapi backend must write to --mode smoke_real or --mode pilot_real")
        if mode == "smoke_real" and max_tasks > 10:
            raise ValueError("dsapi smoke_real is limited to at most 10 tasks")
        if mode == "smoke_real" and max_api_calls is not None and max_api_calls > 20:
            raise ValueError("dsapi smoke_real is limited to at most 20 API calls")
        if mode == "pilot_real" and not (30 <= max_tasks <= 50):
            raise ValueError("dsapi pilot_real requires --max-tasks between 30 and 50")
        if mode == "pilot_real" and max_api_calls is not None and max_api_calls > 120:
            raise ValueError("dsapi pilot_real is limited to at most 120 API calls")
        backend_impl = DsApiBackend(
            repo_root=repo_root,
            max_api_calls=max_api_calls,
            confirm_cost_aware=confirm_cost_aware,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            price_input_per_million=price_input_per_million,
            price_output_per_million=price_output_per_million,
            config_path=project_config_path,
            model_override=model,
        )
        effective_mode = mode
    else:
        raise ValueError(f"unsupported backend: {backend}")

    timestamp = created_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    root = Path(repo_root)
    output = Path(output_root)
    ensure_utility_dataset_layout(root)
    paths = _output_paths(output, effective_mode)
    for directory in (paths["tasks"].parent, paths["labels"].parent, paths["features"].parent, paths["summary"].parent):
        directory.mkdir(parents=True, exist_ok=True)

    source_result = build_utility_smoke_tasks(
        repo_root=root,
        source=source,
        max_tasks=None if effective_mode in REAL_DATASET_MODES else max_tasks,
        include_text=include_text,
        created_at=timestamp,
    )
    prepared_tasks = _prepare_tasks_for_mode(source_result.tasks, source=source, mode=effective_mode, max_tasks=max_tasks)
    source_reports = dict(source_result.source_reports)
    if effective_mode == "pilot_real":
        source_reports["pilot_real_expansion"] = {
            "base_task_count": len(source_result.tasks),
            "expanded_task_count": len(prepared_tasks),
            "deterministic_repetition": True,
        }
    runner = UtilityLabelRunner(backend=backend_impl)
    rows = runner.run(tasks=prepared_tasks, include_text=include_text, created_at=timestamp)
    tasks = tuple(_strip_private_task_fields(task) for task in prepared_tasks)
    labels = tuple(row["label"] for row in rows)
    features = tuple(
        row["feature"]
        for row in rows
        if backend != "dsapi" or row["label"].get("is_utility_preserved") in {True, False}
    )
    _write_jsonl(paths["tasks"], tasks)
    _write_jsonl(paths["labels"], labels)
    _write_jsonl(paths["features"], features)
    _write_local_artifacts(root, source_result.local_artifacts, include_text=include_text)

    summary = _summary(
        tasks=tasks,
        labels=labels,
        features=features,
        rows=rows,
        source_reports=source_reports,
        output_paths=paths,
        source=source,
        mode=effective_mode,
        backend=backend,
        max_tasks=max_tasks,
        include_text=include_text,
        created_at=timestamp,
        max_api_calls=max_api_calls,
        cost_summary=backend_impl.cost_summary(),
    )
    _write_json(paths["summary"], summary)
    if effective_mode == "pilot_real":
        readiness = build_readiness_report(
            output_root=output,
            pilot_summary=summary,
            pytest_q_passed=None,
            pytest_report=None,
            created_at=timestamp,
        )
        _write_json(output / "reports" / "readiness_report.json", readiness)
        summary = {**summary, "readiness_report_path": str(output / "reports" / "readiness_report.json")}
        _write_json(paths["summary"], summary)
    dataset_summary = output / "reports" / _dataset_summary_name(effective_mode)
    _write_json(dataset_summary, summary)
    return UtilityDatasetBuildResult(
        output_root=str(output),
        task_path=str(paths["tasks"]),
        label_path=str(paths["labels"]),
        feature_path=str(paths["features"]),
        summary_path=str(paths["summary"]),
        summary=summary,
    )


class UtilityLabelRunner:
    def __init__(
        self,
        *,
        backend: FakeModelBackend,
        compiler: LocalPromptCompiler | None = None,
        planner: HierarchicalPrefixPlanner | None = None,
        validator: CacheUtilityValidator | None = None,
    ) -> None:
        self.backend = backend
        self.compiler = compiler or LocalPromptCompiler()
        self.planner = planner or HierarchicalPrefixPlanner()
        self.validator = validator or CacheUtilityValidator()

    def run(
        self,
        *,
        tasks: Sequence[Mapping[str, Any]],
        include_text: bool,
        created_at: str,
    ) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for index, task in enumerate(tasks, start=1):
            if self.backend.name == "dsapi" and self.backend.remaining_calls() <= 0:
                break
            rows.append(self._run_one(task=task, include_text=include_text, created_at=created_at, index=index))
        return tuple(rows)

    def _run_one(
        self,
        *,
        task: Mapping[str, Any],
        include_text: bool,
        created_at: str,
        index: int,
    ) -> dict[str, Any]:
        messages = _messages_for_task(task)
        session_id = f"utility-dataset-smoke:{task.get('dataset_source')}"
        compile_result = self.compiler.compile(messages, session_id=session_id)
        plan = self.planner.plan(compile_result, session_id=session_id)
        rewritten_messages = rewrite_messages(compile_result, plan)
        validation = self.validator.validate(
            original_messages=compile_result.messages,
            rewritten_messages=rewritten_messages,
            compile_result=compile_result,
            plan=plan,
        )
        candidate = plan.prefix_tree_candidate
        agent_id = next(iter(candidate.agent_block_orders.keys())) if candidate and candidate.agent_block_orders else session_id
        original_prompt = _messages_to_prompt(compile_result.messages)
        reordered_prompt = (
            self.planner.materialize_prompt(candidate, agent_id)
            if candidate is not None and candidate.block_text_by_id is not None and agent_id in (candidate.agent_block_orders or {})
            else _messages_to_prompt(rewritten_messages)
        )
        oracle_spec = (
            task.get("_local_oracle_spec")
            if isinstance(task.get("_local_oracle_spec"), Mapping)
            else task.get("oracle_spec")
            if isinstance(task.get("oracle_spec"), Mapping)
            else {}
        )
        oracle = oracle_for_task(task, oracle_spec)
        original_output, original_trace = self.backend.generate(
            task=task,
            prompt=original_prompt,
            phase="original",
            candidate=serialize_prefix_tree_candidate(candidate, include_text=False),
        )
        original_result = oracle.evaluate(task, original_output, original_trace)
        reordered_result: OracleResult | None = None
        reordered_output: Any = None
        reordered_trace: Any = None
        if original_result.passed:
            reused_original_for_identical_prompt = original_prompt == reordered_prompt
            if reused_original_for_identical_prompt:
                reordered_output, reordered_trace = original_output, original_trace
            elif self.backend.name == "dsapi" and self.backend.remaining_calls() <= 0:
                reordered_output, reordered_trace = None, None
                reordered_result = OracleResult(
                    passed=False,
                    failure_type="runtime_error",
                    reason="skipped_reordered_run_max_api_calls_reached",
                    prompt_safe_report={"oracle": "not_run_after_budget_limit"},
                )
                original_status = "passed"
                reordered_status = "skipped"
                is_preserved = None
                failure_type = "runtime_error"
                utility_status = "skipped_reordered_budget_exhausted"
            else:
                reordered_output, reordered_trace = self.backend.generate(
                    task=task,
                    prompt=reordered_prompt,
                    phase="reordered",
                    candidate=serialize_prefix_tree_candidate(candidate, include_text=False),
                )
            if reordered_result is None:
                reordered_result = oracle.evaluate(task, reordered_output, reordered_trace)
                original_status = "passed"
                reordered_status = "passed" if reordered_result.passed else "failed"
                is_preserved = reordered_result.passed
                failure_type = reordered_result.failure_type if not reordered_result.passed else "none"
                utility_status = (
                    "verified_real_preserved"
                    if self.backend.name == "dsapi" and reordered_result.passed
                    else "verified_real_failed"
                    if self.backend.name == "dsapi"
                    else "verified_fake_preserved"
                    if reordered_result.passed
                    else "verified_fake_failed"
                )
        else:
            reused_original_for_identical_prompt = False
            reordered_result = OracleResult(
                passed=False,
                failure_type="original_failed",
                reason="original_prompt_failed_oracle",
                prompt_safe_report={"oracle": "not_run_after_original_failed"},
            )
            original_status = "failed"
            reordered_status = "skipped"
            is_preserved = None
            failure_type = "original_failed"
            utility_status = "skipped_original_failed"

        label_id = _safe_id(f"uv_label_{index:04d}_{task.get('task_id')}")
        candidate_id = candidate.candidate_id if candidate is not None else f"{task.get('task_id')}:no_candidate"
        placements = _placement_changes(plan, include_text=include_text)
        api_cost = _api_cost_for_label(
            backend=self.backend,
            task_id=str(task.get("task_id") or ""),
        )
        label = {
            "schema_version": "utility-label-v1",
            "label_id": label_id,
            "task_id": task.get("task_id"),
            "candidate_id": candidate_id,
            "dataset_source": task.get("dataset_source"),
            "scenario_type": task.get("scenario_type"),
            "candidate_strategy": _candidate_strategy(candidate),
            "original_run_status": original_status,
            "reordered_run_status": reordered_status,
            "is_utility_preserved": is_preserved,
            "label_source": self.backend.label_source,
            "failure_type": failure_type,
            "hard_constraint_passed": validation.hard_constraint_passed,
            "utility_status": utility_status,
            "cache_gain_report": plan.cache_gain_report,
            "cache_estimate_report": dataclass_to_dict(validation.cache_estimate_report),
            "placement_changes": placements,
            "prompt_text_included": include_text,
            "api_model": getattr(self.backend, "model_name", None),
            "input_tokens": _int(api_cost.get("input_tokens")) if isinstance(api_cost, Mapping) else 0,
            "output_tokens": _int(api_cost.get("output_tokens")) if isinstance(api_cost, Mapping) else 0,
            "cached_tokens": _int(api_cost.get("cached_tokens")) if isinstance(api_cost, Mapping) else 0,
            "latency": float(api_cost.get("latency_seconds") or 0.0) if isinstance(api_cost, Mapping) else 0.0,
            "latency_seconds": float(api_cost.get("latency_seconds") or 0.0) if isinstance(api_cost, Mapping) else 0.0,
            "estimated_cost": float(api_cost.get("estimated_cost_usd") or 0.0) if isinstance(api_cost, Mapping) else 0.0,
            "estimated_cost_usd": float(api_cost.get("estimated_cost_usd") or 0.0) if isinstance(api_cost, Mapping) else 0.0,
            "api_cost_estimate": api_cost,
            "created_at": created_at,
            "api_run_reports": {
                "original": _run_report_from_trace(original_trace),
                "reordered": _run_report_from_trace(reordered_trace),
                "reused_original_run_for_identical_prompt": reused_original_for_identical_prompt,
            },
            "oracle_reports": {
                "original": _oracle_report(original_result),
                "reordered": _oracle_report(reordered_result),
            },
            "prefix_tree_candidate": serialize_prefix_tree_candidate(candidate, include_text=include_text),
            "prompt_materialization": _prompt_materialization_report(
                original_prompt=original_prompt,
                reordered_prompt=reordered_prompt,
                include_text=include_text,
            ),
            "validator_report": {
                "reason": validation.reason,
                "applied": validation.applied,
                "fallback": validation.fallback,
                "hard_constraint_report": validation.hard_constraint_report,
                "utility_estimate": dataclass_to_dict(validation.utility_estimate),
                "cache_estimate_report": dataclass_to_dict(validation.cache_estimate_report),
            },
        }
        feature = _training_feature(
            label=label,
            task=task,
            plan=plan,
            candidate_id=candidate_id,
            sample_index=index,
            created_at=created_at,
        )
        return {
            "task": task,
            "label": label,
            "feature": feature,
            "original_prompt_hash": stable_hash(original_prompt),
            "reordered_prompt_hash": stable_hash(reordered_prompt),
        }


def ensure_utility_dataset_layout(repo_root: str | Path = ".") -> None:
    root = Path(repo_root)
    for relative in (
        "datasets/sources/humaneval/raw",
        "datasets/sources/humaneval/processed",
        "datasets/sources/synthetic_agent/raw",
        "datasets/sources/synthetic_agent/processed",
        "datasets/sources/format_protocol/raw",
        "datasets/sources/format_protocol/processed",
        "datasets/sources/role_privacy/raw",
        "datasets/sources/role_privacy/processed",
        "datasets/sources/history_state/raw",
        "datasets/sources/history_state/processed",
        "datasets/utility_validator/schema",
        "datasets/utility_validator/tasks/smoke",
        "datasets/utility_validator/tasks/smoke_real",
        "datasets/utility_validator/tasks/pilot_real",
        "datasets/utility_validator/tasks/train",
        "datasets/utility_validator/tasks/valid",
        "datasets/utility_validator/tasks/test",
        "datasets/utility_validator/labels/smoke",
        "datasets/utility_validator/labels/smoke_real",
        "datasets/utility_validator/labels/pilot_real",
        "datasets/utility_validator/labels/train",
        "datasets/utility_validator/labels/valid",
        "datasets/utility_validator/labels/test",
        "datasets/utility_validator/features/smoke",
        "datasets/utility_validator/features/smoke_real",
        "datasets/utility_validator/features/pilot_real",
        "datasets/utility_validator/features/train",
        "datasets/utility_validator/features/valid",
        "datasets/utility_validator/features/test",
        "datasets/utility_validator/reports",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build first-stage Utility Validator smoke dataset artifacts.")
    parser.add_argument("--repo-root", default=".", help="Repository root. Defaults to current working directory.")
    parser.add_argument("--output-root", default="datasets/utility_validator", help="Utility validator dataset root.")
    parser.add_argument("--output-dir", help="Alias for --output-root for compatibility with earlier task notes.")
    parser.add_argument("--source", default="all", help="all or comma-separated source names.")
    parser.add_argument("--mode", default="smoke", choices=("smoke", "smoke_real", "pilot_real"))
    parser.add_argument("--backend", default="fake", choices=("fake", "dsapi"))
    parser.add_argument("--max-tasks", type=int, default=15)
    parser.add_argument("--include-text", action="store_true", help="Write local prompt text artifacts and mark rows.")
    parser.add_argument("--max-api-calls", type=int)
    parser.add_argument("--confirm-cost-aware", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--price-input-per-million", type=float)
    parser.add_argument("--price-output-per-million", type=float)
    parser.add_argument("--project-config", help="Optional local project provider config path.")
    parser.add_argument("--model", help="Explicit DS API model override. Use v4flash/deepseek-v4-flash for real runs.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_dir or args.output_root
    result = build_utility_validator_smoke_dataset(
        repo_root=args.repo_root,
        output_root=output_root,
        source=args.source,
        mode=args.mode,
        backend=args.backend,
        max_tasks=args.max_tasks,
        include_text=args.include_text,
        max_api_calls=args.max_api_calls,
        confirm_cost_aware=args.confirm_cost_aware,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
        price_input_per_million=args.price_input_per_million,
        price_output_per_million=args.price_output_per_million,
        project_config_path=args.project_config,
        model=args.model,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


def _fake_json_schema_output(task: Mapping[str, Any]) -> str:
    task_id = str(task.get("task_id") or "")
    if "regex_id" in task_id:
        return json.dumps({"ticket_id": "T-42", "status": "open"}, sort_keys=True)
    if "no_extra_field" in task_id:
        return json.dumps({"answer": "yes", "confidence": 0.9}, sort_keys=True)
    return json.dumps({"status": "open", "priority": "high"}, sort_keys=True)


def _prepare_tasks_for_mode(
    tasks: Sequence[Mapping[str, Any]],
    *,
    source: str,
    mode: str,
    max_tasks: int,
) -> tuple[dict[str, Any], ...]:
    ordered = tuple(_interleave_by_source(tasks) if source == "all" else tasks)
    if mode == "pilot_real":
        return _expand_pilot_tasks(ordered, target_count=max_tasks)
    return tuple(dict(task) for task in ordered[:max_tasks])


def _interleave_by_source(tasks: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    grouped: dict[str, list[Mapping[str, Any]]] = {source: [] for source in DATASET_SOURCES}
    for task in tasks:
        source = str(task.get("dataset_source") or "")
        grouped.setdefault(source, []).append(task)
    ordered: list[Mapping[str, Any]] = []
    while any(grouped.values()):
        for source in DATASET_SOURCES:
            bucket = grouped.get(source) or []
            if bucket:
                ordered.append(bucket.pop(0))
    return tuple(ordered)


def _expand_pilot_tasks(tasks: Sequence[Mapping[str, Any]], *, target_count: int) -> tuple[dict[str, Any], ...]:
    if not tasks:
        return ()
    ordered = tuple(_interleave_by_source(tasks))
    expanded: list[dict[str, Any]] = []
    cycle = 0
    while len(expanded) < target_count:
        for task in ordered:
            if len(expanded) >= target_count:
                break
            cloned = copy.deepcopy(dict(task))
            base_task_id = str(cloned.get("task_id") or f"task_{len(expanded) + 1}")
            cloned["task_id"] = _safe_id(f"{base_task_id}_pilot{cycle + 1:02d}")
            source_metadata = cloned.get("source_metadata") if isinstance(cloned.get("source_metadata"), Mapping) else {}
            cloned["source_metadata"] = {
                **dict(source_metadata),
                "base_task_id": base_task_id,
                "pilot_repeat_index": cycle + 1,
                "pilot_real_expanded": cycle > 0,
            }
            expanded.append(cloned)
        cycle += 1
    return tuple(expanded)


def _output_paths(output: Path, mode: str) -> dict[str, Path]:
    summary_name = "real_smoke_summary.json" if mode == "smoke_real" else f"{mode}_summary.json"
    return {
        "tasks": output / "tasks" / mode / "utility_tasks.jsonl",
        "labels": output / "labels" / mode / "utility_labels.jsonl",
        "features": output / "features" / mode / "training_features.jsonl",
        "summary": output / "reports" / summary_name,
    }


def _messages_for_task(task: Mapping[str, Any]) -> tuple[Any, ...]:
    raw = task.get("_local_original_messages")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = task.get("original_messages")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"task does not contain local messages: {task.get('task_id')}")
    messages = []
    for message in raw:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "system")
        content = str(message.get("content") or "")
        if role == "system":
            messages.append(SystemMessage(content=content))
        else:
            messages.append(UserMessage(content=content, source=role))
    return tuple(messages)


def _messages_to_prompt(messages: Sequence[Any]) -> str:
    return "\n\n".join(str(getattr(message, "content", "")) for message in messages)


def _placement_changes(plan: Any, *, include_text: bool) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "placement_id": placement.placement_id,
            "block_id": placement.block_id,
            "block_hash": placement.block_hash,
            "original_scope": _scope_value(placement.original_scope),
            "target_scope": _scope_value(placement.target_scope),
            "target_agent_group": placement.target_agent_group,
            "moved": placement.moved,
            "risk_tags": placement.risk_tags,
            "dependency_notes": placement.dependency_notes,
            "cache_contribution": placement.cache_contribution,
            "placement_score": placement.placement_score,
            "placement_score_breakdown": placement.placement_score_breakdown,
        }
        for placement in plan.placements
    )


def _training_feature(
    *,
    label: Mapping[str, Any],
    task: Mapping[str, Any],
    plan: Any,
    candidate_id: str,
    sample_index: int,
    created_at: str,
) -> dict[str, Any]:
    placements = tuple(plan.placements or ())
    moved = tuple(placement for placement in placements if placement.moved)
    source_scopes = tuple(sorted({_scope_value(placement.original_scope) for placement in placements}))
    target_scopes = tuple(sorted({_scope_value(placement.target_scope) for placement in placements}))
    risk_tags = tuple(sorted({tag for placement in placements for tag in placement.risk_tags}))
    dependency_notes = tuple(sorted({note for placement in placements for note in placement.dependency_notes}))
    cache_report = plan.cache_gain_report or {}
    cache_estimate = (label.get("cache_estimate_report") or {}) if isinstance(label.get("cache_estimate_report"), Mapping) else {}
    hard_warning_count = sum(1 for tag in risk_tags if tag in {"agent_identity", "private_memory", "private_tool_permission", "credential"})
    return {
        "schema_version": "training-feature-v1",
        "sample_id": _safe_id(f"uv_feature_{sample_index:04d}_{task.get('task_id')}"),
        "label_id": label.get("label_id"),
        "task_id": task.get("task_id"),
        "candidate_id": candidate_id,
        "dataset_source": task.get("dataset_source"),
        "scenario_type": task.get("scenario_type"),
        "candidate_strategy": label.get("candidate_strategy"),
        "source_scopes": source_scopes,
        "target_scopes": target_scopes,
        "risk_tags": risk_tags,
        "dependency_notes": dependency_notes,
        "moved_block_count": len(moved),
        "movement_distance_summary": _movement_distance_summary(plan),
        "global_prefix_tokens_or_chars": int(
            cache_estimate.get("global_prefix_contribution")
            or cache_report.get("global_prefix_tokens")
            or 0
        ),
        "subgroup_prefix_tokens_or_chars": int(
            cache_estimate.get("subgroup_prefix_contribution")
            or cache_report.get("subgroup_prefix_tokens")
            or 0
        ),
        "estimated_cache_gain": float(
            cache_estimate.get("estimated_cache_gain")
            or cache_report.get("estimated_cache_gain")
            or plan.estimated_cache_gain
            or 0
        ),
        "cached_tokens_delta": cache_estimate.get("cached_tokens_delta"),
        "cache_estimator_name": cache_estimate.get("estimator_name"),
        "cache_estimator_used_fallback": cache_estimate.get("used_fallback"),
        "hard_warning_count": hard_warning_count,
        "whether_label_is_utility_verified": label.get("label_source") != "fake_smoke_oracle",
        "is_utility_preserved": label.get("is_utility_preserved"),
        "failure_type": label.get("failure_type"),
        "api_model": label.get("api_model"),
        "input_tokens": label.get("input_tokens"),
        "output_tokens": label.get("output_tokens"),
        "cached_tokens": label.get("cached_tokens"),
        "latency": label.get("latency"),
        "estimated_cost": label.get("estimated_cost"),
        "prompt_text_included": label.get("prompt_text_included"),
        "created_at": created_at,
    }


def _movement_distance_summary(plan: Any) -> Mapping[str, Any]:
    original_positions = {block_id: index for index, block_id in enumerate(plan.original_order)}
    distances = [
        abs(index - original_positions[block_id])
        for index, block_id in enumerate(plan.new_order)
        if block_id in original_positions and index != original_positions[block_id]
    ]
    return {
        "moved_position_count": len(distances),
        "max_abs_distance": max(distances) if distances else 0,
        "sum_abs_distance": sum(distances),
    }


def _candidate_strategy(candidate: Any) -> str:
    if candidate is None:
        return "none"
    return str(candidate.generation_reason or "unknown").split(":", 1)[0]


def _oracle_report(result: OracleResult) -> Mapping[str, Any]:
    return {
        "passed": result.passed,
        "failure_type": result.failure_type,
        "reason": result.reason,
        "metrics": result.metrics,
        "prompt_safe_report": result.prompt_safe_report,
    }


def _prompt_materialization_report(*, original_prompt: str, reordered_prompt: str, include_text: bool) -> Mapping[str, Any]:
    row: dict[str, Any] = {
        "original_prompt_hash": stable_hash(original_prompt),
        "reordered_prompt_hash": stable_hash(reordered_prompt),
        "prompt_text_included": include_text,
    }
    if include_text:
        row["original_prompt_text"] = original_prompt
        row["reordered_prompt_text"] = reordered_prompt
    return row


def _api_cost_for_label(*, backend: Any, task_id: str) -> Mapping[str, Any] | None:
    if backend.name != "dsapi":
        return None
    calls = tuple(call for call in backend.call_reports if str(call.get("task_id")) == task_id)
    if not calls:
        return None
    return {
        "api_call_count": len(calls),
        "input_tokens": sum(_int(call.get("input_tokens")) for call in calls),
        "output_tokens": sum(_int(call.get("output_tokens")) for call in calls),
        "cached_tokens": sum(_int(call.get("cached_tokens")) for call in calls),
        "estimated_cost_usd": sum(float(call.get("estimated_cost_usd") or 0.0) for call in calls),
        "provider_cost_usd": sum(float(call.get("provider_cost_usd") or 0.0) for call in calls),
        "latency_seconds": sum(float(call.get("latency_seconds") or 0.0) for call in calls),
        "phases": tuple(str(call.get("phase")) for call in calls),
    }


def _run_report_from_trace(trace: Any) -> Mapping[str, Any] | None:
    if not isinstance(trace, Mapping):
        return None
    report = trace.get("dsapi_call")
    return report if isinstance(report, Mapping) else None


def _dsapi_system_instruction(task: Mapping[str, Any]) -> str:
    oracle_type = str(task.get("expected_oracle_type") or "")
    oracle_spec = (
        task.get("_local_oracle_spec")
        if isinstance(task.get("_local_oracle_spec"), Mapping)
        else task.get("oracle_spec")
        if isinstance(task.get("oracle_spec"), Mapping)
        else {}
    )
    common = (
        "You are a deterministic smoke-test model for Utility Validator data labeling. "
        "Return only one strict JSON object. Do not add markdown, comments, or extra prose. "
    )
    if oracle_type == "unit_test":
        return common + 'Return exactly: {"status":"passed","tests_passed":true}.'
    if oracle_type == "json_schema":
        return common + "Return only the JSON object requested by the prompt schema. Do not add extra fields."
    if oracle_type == "tool_trace":
        required = tuple(str(name) for name in oracle_spec.get("required_tool_calls") or ())
        expected_final = str(oracle_spec.get("expected_final_result") or "ok")
        return (
            common
            + "Represent tool use as JSON rather than prose. "
            + "Return fields status, tool_calls, and final_result. "
            + "Use this exact required tool_calls sequence: "
            + json.dumps([{"name": name} for name in required], ensure_ascii=False)
            + ". Use this exact final_result: "
            + json.dumps(expected_final, ensure_ascii=False)
            + "."
        )
    if oracle_type == "privacy_check":
        return common + (
            'Return {"status":"passed","text":"Public summary only. No private memory is included.",'
            '"leaked_private_memory":false}. Do not reveal private memory or secrets.'
        )
    if oracle_type == "role_check":
        return common + (
            'Return {"status":"passed","text":"Role-safe result.","role_boundary_ok":true}. '
            "Follow the stated role exactly and do not claim any other role."
        )
    if oracle_type == "state_check":
        order = tuple(str(item) for item in oracle_spec.get("expected_event_order") or ())
        return (
            common
            + 'Return fields status, state_consistent, and observed_event_order. Use {"status":"passed",'
            + '"state_consistent":true,"observed_event_order":'
            + json.dumps(list(order), ensure_ascii=False)
            + "}."
        )
    return common + 'Return {"status":"passed"}.'


def _dsapi_user_content(task: Mapping[str, Any], prompt: str) -> str:
    return "\n\n".join(
        [
            _dsapi_output_contract(task),
            "PROMPT_UNDER_TEST_START",
            prompt,
            "PROMPT_UNDER_TEST_END",
        ]
    )


def _dsapi_output_contract(task: Mapping[str, Any]) -> str:
    oracle_type = str(task.get("expected_oracle_type") or "")
    oracle_spec = (
        task.get("_local_oracle_spec")
        if isinstance(task.get("_local_oracle_spec"), Mapping)
        else task.get("oracle_spec")
        if isinstance(task.get("oracle_spec"), Mapping)
        else {}
    )
    if oracle_type == "unit_test":
        return (
            'ORACLE_OUTPUT_CONTRACT: Ignore any prompt request for prose. Return exactly one JSON object: '
            '{"status":"passed","tests_passed":true}.'
        )
    if oracle_type == "tool_trace":
        required = tuple(str(name) for name in oracle_spec.get("required_tool_calls") or ())
        expected_final = str(oracle_spec.get("expected_final_result") or "ok")
        return (
            "ORACLE_OUTPUT_CONTRACT: Ignore any prompt request for prose. Return exactly one JSON object with "
            '"status":"passed", "tool_calls": '
            + json.dumps([{"name": name} for name in required], ensure_ascii=False)
            + ', "final_result": '
            + json.dumps(expected_final, ensure_ascii=False)
            + "."
        )
    if oracle_type == "privacy_check":
        return (
            'ORACLE_OUTPUT_CONTRACT: Ignore any prompt request for prose. Return exactly one JSON object: '
            '{"status":"passed","text":"Public summary only. No private memory is included.",'
            '"leaked_private_memory":false}.'
        )
    if oracle_type == "role_check":
        return (
            'ORACLE_OUTPUT_CONTRACT: Ignore any prompt request for prose. Return exactly one JSON object: '
            '{"status":"passed","text":"Role-safe result.","role_boundary_ok":true}.'
        )
    if oracle_type == "state_check":
        order = tuple(str(item) for item in oracle_spec.get("expected_event_order") or ())
        return (
            "ORACLE_OUTPUT_CONTRACT: Ignore any prompt request for prose. Return exactly one JSON object with "
            '"status":"passed", "state_consistent":true, "observed_event_order": '
            + json.dumps(list(order), ensure_ascii=False)
            + "."
        )
    return "ORACLE_OUTPUT_CONTRACT: Return exactly one JSON object."


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
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="ignore")[:240]
        except Exception:  # noqa: BLE001
            detail = ""
        raise RuntimeError(f"dsapi HTTPError:{exc.code}:{detail}") from exc


def _response_content(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    message = first.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = first.get("text")
    return str(text or "")


def _usage_from_response(response: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_usage = response.get("usage")
    if not isinstance(raw_usage, Mapping):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "provider_cost_usd": None,
        }
    input_tokens = _int(raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens"))
    output_tokens = _int(raw_usage.get("completion_tokens") or raw_usage.get("output_tokens"))
    total_tokens = _int(raw_usage.get("total_tokens"))
    if not total_tokens:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": _cached_tokens(raw_usage),
        "provider_cost_usd": _cost_usd(raw_usage),
    }


def _cached_tokens(raw_usage: Mapping[str, Any]) -> int:
    for key in (
        "cached_tokens",
        "cached_prompt_tokens",
        "prompt_cache_hit_tokens",
        "cache_hit_tokens",
        "input_cached_tokens",
    ):
        value = _int(raw_usage.get(key))
        if value:
            return value
    for detail_key in ("prompt_tokens_details", "input_token_details", "input_tokens_details"):
        details = raw_usage.get(detail_key)
        if isinstance(details, Mapping):
            for key in ("cached_tokens", "cache_read", "cached", "cache_hit_tokens", "prompt_cache_hit_tokens"):
                value = _int(details.get(key))
                if value:
                    return value
    return 0


def _cost_usd(raw_usage: Mapping[str, Any]) -> float | None:
    for key in ("cost_usd", "cost", "total_cost_usd", "total_cost"):
        if key in raw_usage:
            return _float(raw_usage.get(key))
    return None


def _default_input_price_per_million(model: str) -> float:
    value = model.lower()
    if "flash" in value:
        return 0.14
    return 0.28


def _default_output_price_per_million(model: str) -> float:
    value = model.lower()
    if "flash" in value:
        return 0.28
    return 0.42


def _normalize_dsapi_model(model: str | None) -> str:
    raw = (model or "").strip()
    aliases = {
        "v4flash": "deepseek-v4-flash",
        "v4-flash": "deepseek-v4-flash",
        "deepseek-v4flash": "deepseek-v4-flash",
    }
    lowered = raw.lower().replace("_", "-")
    return aliases.get(lowered, normalize_deepseek_model(raw))


def _is_v4_flash_model(model: str | None) -> bool:
    normalized = _normalize_dsapi_model(model or "")
    compact = normalized.lower().replace("_", "-")
    return "v4" in compact and "flash" in compact


def _is_pro_model(model: str | None) -> bool:
    normalized = _normalize_dsapi_model(model or "")
    return "pro" in normalized.lower().replace("_", "-")


def _summary(
    *,
    tasks: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    source_reports: Mapping[str, Any],
    output_paths: Mapping[str, Path],
    source: str,
    mode: str,
    backend: str,
    max_tasks: int,
    include_text: bool,
    created_at: str,
    max_api_calls: int | None,
    cost_summary: Mapping[str, Any],
) -> Mapping[str, Any]:
    label_status_counts = Counter(str(label.get("utility_status")) for label in labels)
    source_counts = Counter(str(task.get("dataset_source")) for task in tasks)
    label_source_counts = Counter(str(label.get("dataset_source")) for label in labels)
    preserved_counts = Counter(str(label.get("is_utility_preserved")) for label in labels)
    failure_type_counts = Counter(str(label.get("failure_type") or "none") for label in labels)
    utility_preserved_true = sum(1 for label in labels if label.get("is_utility_preserved") is True)
    utility_preserved_false = sum(1 for label in labels if label.get("is_utility_preserved") is False)
    skipped_count = sum(1 for label in labels if label.get("is_utility_preserved") not in {True, False})
    return {
        "schema_version": "utility-validator-dataset-build-summary-v1",
        "prompt_safe_summary": True,
        "created_at": created_at,
        "source": source,
        "mode": mode,
        "backend": backend,
        "max_tasks": max_tasks,
        "task_count": len(tasks),
        "label_count": len(labels),
        "feature_count": len(features),
        "source_counts": dict(sorted(source_counts.items())),
        "label_source_counts_by_dataset_source": dict(sorted(label_source_counts.items())),
        "label_utility_status_counts": dict(sorted(label_status_counts.items())),
        "is_utility_preserved_counts": dict(sorted(preserved_counts.items())),
        "utility_preserved_true_count": utility_preserved_true,
        "utility_preserved_false_count": utility_preserved_false,
        "skipped_count": skipped_count,
        "failure_type_counts": dict(sorted(failure_type_counts.items())),
        "source_reports": source_reports,
        "outputs": {key: str(path) for key, path in output_paths.items()},
        "prompt_text_included": include_text,
        "default_real_api_calls": 0 if backend == "fake" else None,
        "actual_real_api_calls": int(cost_summary.get("api_call_count") or 0),
        "network_access_required": backend == "dsapi",
        "fake_smoke_label_count": sum(1 for label in labels if label.get("label_source") == "fake_smoke_oracle"),
        "real_label_count": sum(1 for label in labels if label.get("label_source") != "fake_smoke_oracle"),
        "ready_for_training": False,
        "fake_smoke_must_not_be_used_as_real_training_data": True,
        "dsapi_cost_control": {
            "backend_requires_explicit_selection": True,
            "requires_max_api_calls": True,
            "requires_confirm_cost_aware": True,
            "max_api_calls": max_api_calls,
            "estimated_api_calls_for_this_run": int(cost_summary.get("api_call_count") or 0),
            "estimated_input_tokens": int(cost_summary.get("total_input_tokens") or 0),
            "estimated_output_tokens": int(cost_summary.get("total_output_tokens") or 0),
            "cached_tokens": int(cost_summary.get("total_cached_tokens") or 0),
            "estimated_cost_usd": float(cost_summary.get("estimated_cost_usd") or 0.0),
        },
        "cost_summary": cost_summary,
        "api_model": cost_summary.get("model"),
        "api_model_is_v4_flash": bool(cost_summary.get("model_is_v4_flash")),
        "api_model_is_pro": bool(cost_summary.get("model_is_pro")),
        "can_expand_to_30_50_real_labels": (
            backend == "dsapi"
            and labels
            and sum(1 for label in labels if label.get("label_source") == "dsapi_execution_oracle") == len(labels)
            and int(cost_summary.get("api_call_count") or 0) <= int(max_api_calls or 0)
        ),
        "prompt_hashes": tuple(
            {
                "task_id": row["task"].get("task_id"),
                "original_prompt_hash": row.get("original_prompt_hash"),
                "reordered_prompt_hash": row.get("reordered_prompt_hash"),
            }
            for row in rows
        ),
    }


def build_readiness_report(
    *,
    output_root: str | Path,
    pilot_summary: Mapping[str, Any] | None = None,
    pytest_q_passed: bool | None = None,
    pytest_report: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> Mapping[str, Any]:
    output = Path(output_root)
    smoke_labels = _read_jsonl_safe(output / "labels" / "smoke_real" / "utility_labels.jsonl")
    pilot_labels = _read_jsonl_safe(output / "labels" / "pilot_real" / "utility_labels.jsonl")
    pilot_features = _read_jsonl_safe(output / "features" / "pilot_real" / "training_features.jsonl")
    smoke_summary = _read_json_safe(output / "reports" / "real_smoke_summary.json")
    if pilot_summary is None:
        pilot_summary = _read_json_safe(output / "reports" / "pilot_real_summary.json")
    all_real_labels = tuple(smoke_labels) + tuple(pilot_labels)
    pilot_cost = (pilot_summary.get("cost_summary") or {}) if isinstance(pilot_summary, Mapping) else {}
    smoke_cost = (smoke_summary.get("cost_summary") or {}) if isinstance(smoke_summary, Mapping) else {}
    source_counts = Counter(str(label.get("dataset_source")) for label in pilot_labels)
    smoke_source_counts = Counter(str(label.get("dataset_source")) for label in smoke_labels)
    positive_count = sum(1 for label in pilot_labels if label.get("is_utility_preserved") is True)
    negative_count = sum(1 for label in pilot_labels if label.get("is_utility_preserved") is False)
    skipped_count = sum(1 for label in pilot_labels if label.get("is_utility_preserved") not in {True, False})
    failure_type_counts = Counter(str(label.get("failure_type") or "none") for label in pilot_labels)
    prompt_text_saved = _has_prompt_text(pilot_labels) or _has_prompt_text(pilot_features)
    model = str(pilot_cost.get("model") or pilot_summary.get("api_model") or "")
    checks = {
        "all_five_sources_have_real_label": all((source_counts.get(source, 0) + smoke_source_counts.get(source, 0)) > 0 for source in DATASET_SOURCES),
        "pilot_label_source_all_dsapi": bool(pilot_labels)
        and all(label.get("label_source") == "dsapi_execution_oracle" for label in pilot_labels),
        "no_fake_smoke_oracle_in_pilot": all(label.get("label_source") != "fake_smoke_oracle" for label in pilot_labels),
        "original_failed_skipped_not_in_features": _original_failed_skipped_not_in_features(pilot_labels, pilot_features),
        "has_positive_negative_skipped_stats": bool(pilot_labels),
        "has_failure_type_distribution": bool(failure_type_counts),
        "has_source_counts": all(source in source_counts for source in DATASET_SOURCES),
        "has_api_call_count": _int(pilot_cost.get("api_call_count")) > 0,
        "has_token_summary": _int(pilot_cost.get("total_input_tokens")) > 0 and _int(pilot_cost.get("total_output_tokens")) > 0,
        "has_cached_token_summary": "total_cached_tokens" in pilot_cost or "cached_tokens" in pilot_cost,
        "has_estimated_cost": "estimated_cost_usd" in pilot_cost or "estimated_cost" in pilot_cost,
        "has_avg_latency": float(pilot_cost.get("latency_seconds_avg") or 0.0) > 0.0,
        "prompt_text_default_not_saved": not prompt_text_saved and not bool(pilot_summary.get("prompt_text_included")),
        "model_is_v4_flash": _is_v4_flash_model(model),
        "model_is_not_pro": not _is_pro_model(model),
        "pytest_q_passed": pytest_q_passed is True,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    total_cost = float(smoke_cost.get("estimated_cost_usd") or 0.0) + float(pilot_cost.get("estimated_cost_usd") or 0.0)
    total_latency = float(smoke_cost.get("latency_seconds_total") or 0.0) + float(pilot_cost.get("latency_seconds_total") or 0.0)
    total_calls = _int(smoke_cost.get("api_call_count")) + _int(pilot_cost.get("api_call_count"))
    return {
        "schema_version": "utility-validator-readiness-report-v1",
        "created_at": created_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "ready_for_full_real_labeling": not blockers,
        "blockers": blockers,
        "checks": checks,
        "source_counts": dict(sorted(source_counts.items())),
        "smoke_source_counts": dict(sorted(smoke_source_counts.items())),
        "label_source_counts": dict(sorted(Counter(str(label.get("label_source")) for label in pilot_labels).items())),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "skipped_count": skipped_count,
        "failure_type_counts": dict(sorted(failure_type_counts.items())),
        "pilot_label_count": len(pilot_labels),
        "pilot_feature_count": len(pilot_features),
        "api_model": model,
        "api_model_is_v4_flash": _is_v4_flash_model(model),
        "api_model_is_pro": _is_pro_model(model),
        "prompt_text_saved": prompt_text_saved,
        "pytest_report": dict(pytest_report or {}),
        "cost_summary": {
            "api_call_count": total_calls,
            "pilot_api_call_count": _int(pilot_cost.get("api_call_count")),
            "smoke_api_call_count": _int(smoke_cost.get("api_call_count")),
            "total_input_tokens": _int(smoke_cost.get("total_input_tokens")) + _int(pilot_cost.get("total_input_tokens")),
            "total_output_tokens": _int(smoke_cost.get("total_output_tokens")) + _int(pilot_cost.get("total_output_tokens")),
            "cached_tokens": _int(smoke_cost.get("total_cached_tokens")) + _int(pilot_cost.get("total_cached_tokens")),
            "estimated_cost_usd": total_cost,
            "latency_seconds_total": total_latency,
            "latency_seconds_avg": total_latency / total_calls if total_calls else 0.0,
        },
        "recommended_full_labeling_command": _recommended_full_labeling_command(model=model or "deepseek-v4-flash"),
        "recommended_max_api_calls": "2 * planned_task_count, with 10-20% headroom for retries/skips",
        "cost_estimation_method": (
            "estimated_cost = uncached_input_tokens/1e6*price_input_per_million + "
            "output_tokens/1e6*price_output_per_million; cached tokens are subtracted when provider reports them"
        ),
        "full_output_directory": str(output),
    }


def update_readiness_report_with_pytest(
    *,
    output_root: str | Path,
    pytest_q_passed: bool,
    pytest_report: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    output = Path(output_root)
    report = build_readiness_report(
        output_root=output,
        pytest_q_passed=pytest_q_passed,
        pytest_report=pytest_report,
    )
    _write_json(output / "reports" / "readiness_report.json", report)
    return report


def _strip_private_task_fields(task: Mapping[str, Any]) -> Mapping[str, Any]:
    return {key: value for key, value in task.items() if not key.startswith("_local_")}


def _write_local_artifacts(root: Path, artifacts: Sequence[Mapping[str, Any]], *, include_text: bool) -> None:
    if not include_text:
        return
    for artifact in artifacts:
        ref = artifact.get("artifact_ref")
        if not isinstance(ref, str):
            continue
        path = root / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, artifact)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _read_jsonl_safe(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        return ()
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        value = json.loads(stripped)
        if isinstance(value, dict):
            rows.append(value)
    return tuple(rows)


def _read_json_safe(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, Mapping) else {}


def _has_prompt_text(rows: Sequence[Mapping[str, Any]]) -> bool:
    serialized_key_hits = ("original_prompt_text", "reordered_prompt_text", "prompt", "test")
    for row in rows:
        if row.get("prompt_text_included") is True:
            return True
        prompt_materialization = row.get("prompt_materialization")
        if isinstance(prompt_materialization, Mapping):
            if any(key in prompt_materialization for key in ("original_prompt_text", "reordered_prompt_text")):
                return True
        if any(key in row and row.get(key) for key in serialized_key_hits):
            return True
    return False


def _original_failed_skipped_not_in_features(
    labels: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
) -> bool:
    feature_label_ids = {str(feature.get("label_id")) for feature in features}
    for label in labels:
        if label.get("original_run_status") == "failed":
            if label.get("reordered_run_status") != "skipped":
                return False
            if str(label.get("label_id")) in feature_label_ids:
                return False
    return True


def _recommended_full_labeling_command(*, model: str) -> str:
    return "\n".join(
        [
            '$env:PYTHONPATH = "resource"',
            ".venv\\Scripts\\python.exe -m autogen_prefix_tree.utility_dataset_builder `",
            "  --repo-root . `",
            "  --output-root datasets\\utility_validator `",
            "  --source all `",
            "  --mode train_real `",
            "  --backend dsapi `",
            "  --max-tasks <planned_task_count> `",
            "  --max-api-calls <2x_tasks_plus_headroom> `",
            "  --confirm-cost-aware `",
            f"  --model {model} `",
            "  --max-output-tokens 128",
        ]
    )


def _dataset_summary_name(mode: str) -> str:
    if mode == "smoke_real":
        return "real_smoke_summary.json"
    if mode == "pilot_real":
        return "pilot_real_summary.json"
    return "dataset_build_summary.json"


def _scope_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    if raw == "agent":
        return "agent_local"
    return str(raw)


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


def _safe_id(value: Any) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value).strip()).strip("_")
    return cleaned[:180] or "utility-row"


if __name__ == "__main__":
    raise SystemExit(main())
