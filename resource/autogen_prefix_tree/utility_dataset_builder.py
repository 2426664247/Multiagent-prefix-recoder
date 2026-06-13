from __future__ import annotations

import argparse
import copy
import json
import re
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
from .ir import Movability, SemanticType, ShareScope, stable_hash
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


@dataclass(frozen=True)
class FullUtilityDatasetBuildResult:
    output_root: str
    summary_path: str
    cost_summary_path: str
    quality_report_path: str
    dataset_card_path: str
    training_usage_path: str
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class ExpandedUtilityDatasetBuildResult:
    output_root: str
    summary_path: str
    cost_summary_path: str
    quality_report_path: str
    dataset_card_path: str
    training_usage_path: str
    summary: Mapping[str, Any]


REAL_DATASET_MODES = {"smoke_real", "pilot_real", "train_real", "expanded_real"}
FINAL_SPLITS = ("train", "valid", "test")
EXPANDED_SPLITS = ("expanded_train", "expanded_valid", "expanded_test")
EXPANDED_STRATEGIES = ("conservative", "balanced", "aggressive", "adversarial")


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
    if mode == "expanded_real":
        result = build_expanded_utility_validator_dataset(
            repo_root=repo_root,
            output_root=output_root,
            source=source,
            backend=backend,
            max_tasks=max_tasks,
            include_text=include_text,
            max_api_calls=max_api_calls,
            confirm_cost_aware=confirm_cost_aware,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            price_input_per_million=price_input_per_million,
            price_output_per_million=price_output_per_million,
            project_config_path=project_config_path,
            model=model,
            created_at=created_at,
        )
        return UtilityDatasetBuildResult(
            output_root=result.output_root,
            task_path="",
            label_path="",
            feature_path="",
            summary_path=result.summary_path,
            summary=result.summary,
        )
    if mode == "train_real":
        result = build_full_utility_validator_dataset(
            repo_root=repo_root,
            output_root=output_root,
            source=source,
            backend=backend,
            max_tasks=max_tasks,
            include_text=include_text,
            max_api_calls=max_api_calls,
            confirm_cost_aware=confirm_cost_aware,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            price_input_per_million=price_input_per_million,
            price_output_per_million=price_output_per_million,
            project_config_path=project_config_path,
            model=model,
            created_at=created_at,
        )
        return UtilityDatasetBuildResult(
            output_root=result.output_root,
            task_path="",
            label_path="",
            feature_path="",
            summary_path=result.summary_path,
            summary=result.summary,
        )
    if mode not in {"smoke", "smoke_real", "pilot_real"}:
        raise ValueError("only smoke, smoke_real, pilot_real, train_real, and expanded_real modes are implemented in the dataset builder")
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


def build_full_utility_validator_dataset(
    *,
    repo_root: str | Path = ".",
    output_root: str | Path = "datasets/utility_validator",
    source: str = "all",
    backend: str = "dsapi",
    max_tasks: int = 50,
    include_text: bool = False,
    max_api_calls: int | None = None,
    confirm_cost_aware: bool = False,
    max_output_tokens: int = 128,
    timeout_seconds: float = 120.0,
    price_input_per_million: float | None = None,
    price_output_per_million: float | None = None,
    project_config_path: str | Path | None = None,
    model: str | None = None,
    batch_size: int = 25,
    checkpoint_api_calls: int = 100,
    max_estimated_cost_usd: float = 1.0,
    created_at: str | None = None,
) -> FullUtilityDatasetBuildResult:
    if backend != "dsapi":
        raise ValueError("train_real requires --backend dsapi")
    if not confirm_cost_aware or max_api_calls is None or max_api_calls <= 0:
        raise ValueError("train_real requires --max-api-calls N --confirm-cost-aware")
    if include_text:
        raise ValueError("train_real refuses --include-text by default for prompt-safe training data")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if checkpoint_api_calls <= 0:
        raise ValueError("--checkpoint-api-calls must be positive")
    if max_tasks <= 0:
        raise ValueError("--max-tasks must be positive")

    timestamp = created_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    root = Path(repo_root)
    output = Path(output_root)
    ensure_utility_dataset_layout(root)
    full_paths = _full_output_paths(output)
    _prepare_full_output_dirs(full_paths)
    checkpoint_root = output / "reports" / "full_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

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
    if backend_impl.model_name != "deepseek-v4-flash":
        raise ValueError(
            f"train_real requires exact model deepseek-v4-flash; effective model is {backend_impl.model_name!r}"
        )

    source_result = build_utility_smoke_tasks(
        repo_root=root,
        source=source,
        max_tasks=None,
        include_text=False,
        created_at=timestamp,
    )
    base_tasks = _prepare_tasks_for_mode(source_result.tasks, source=source, mode="pilot_real", max_tasks=max_tasks)
    batches = _batch_tasks(base_tasks, batch_size=batch_size)
    all_tasks: list[dict[str, Any]] = []
    all_labels: list[dict[str, Any]] = []
    all_features: list[dict[str, Any]] = []
    checkpoint_reports: list[dict[str, Any]] = []
    anomalous_batches: list[Mapping[str, Any]] = []
    total_cost_at_last_checkpoint = 0.0
    started_at = time.perf_counter()

    runner = UtilityLabelRunner(backend=backend_impl)
    for batch_index, batch_tasks in enumerate(batches, start=1):
        calls_before = backend_impl.calls_made
        cost_before = backend_impl.cost_summary()
        rows = runner.run(tasks=batch_tasks, include_text=False, created_at=timestamp)
        labels = tuple(row["label"] for row in rows)
        features = tuple(row["feature"] for row in rows if row["label"].get("is_utility_preserved") in {True, False})
        tasks = tuple(_strip_private_task_fields(task) for task in batch_tasks[: len(rows)])
        cost_after = backend_impl.cost_summary()
        checkpoint_report = _checkpoint_report(
            batch_index=batch_index,
            tasks=tasks,
            labels=labels,
            features=features,
            cost_before=cost_before,
            cost_after=cost_after,
            max_estimated_cost_usd=max_estimated_cost_usd,
            max_api_calls=max_api_calls,
            checkpoint_api_calls=checkpoint_api_calls,
            output_root=output,
            created_at=timestamp,
        )
        batch_dir = checkpoint_root / f"batch_{batch_index:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(batch_dir / "utility_tasks.jsonl", tasks)
        _write_jsonl(batch_dir / "utility_labels.jsonl", labels)
        _write_jsonl(batch_dir / "training_features.jsonl", features)
        _write_json(batch_dir / "checkpoint_report.json", checkpoint_report)
        checkpoint_reports.append(checkpoint_report)
        if not checkpoint_report["passed"]:
            anomalous_batches.append(
                {
                    "batch_index": batch_index,
                    "report_path": str(batch_dir / "checkpoint_report.json"),
                    "blockers": checkpoint_report["blockers"],
                    "handled": "paused_before_merge",
                }
            )
            break
        all_tasks.extend(dict(task) for task in tasks)
        all_labels.extend(dict(label) for label in labels)
        all_features.extend(dict(feature) for feature in features)
        total_cost_at_last_checkpoint = float(cost_after.get("estimated_cost_usd") or 0.0)
        if backend_impl.calls_made >= max_api_calls:
            break
        if backend_impl.calls_made == calls_before and rows:
            raise RuntimeError("checkpoint made no API-call progress despite emitted rows")

    split_rows = _split_full_rows(all_tasks, all_labels, all_features)
    _write_split_outputs(full_paths, split_rows)
    cost_summary = _full_cost_summary(backend_impl.cost_summary())
    quality_report = _full_quality_report(
        split_rows=split_rows,
        checkpoint_reports=checkpoint_reports,
        anomalous_batches=anomalous_batches,
        output_root=output,
        created_at=timestamp,
    )
    elapsed_seconds = time.perf_counter() - started_at
    full_summary = _full_build_summary(
        source=source,
        max_tasks=max_tasks,
        batch_size=batch_size,
        checkpoint_api_calls=checkpoint_api_calls,
        max_api_calls=max_api_calls,
        include_text=include_text,
        created_at=timestamp,
        elapsed_seconds=elapsed_seconds,
        split_rows=split_rows,
        cost_summary=cost_summary,
        quality_report=quality_report,
        checkpoint_reports=checkpoint_reports,
        anomalous_batches=anomalous_batches,
        output_paths=full_paths,
        source_reports=source_result.source_reports,
        total_cost_at_last_checkpoint=total_cost_at_last_checkpoint,
    )
    _write_json(full_paths["reports"]["cost_summary"], cost_summary)
    _write_json(full_paths["reports"]["quality_report"], quality_report)
    _write_json(full_paths["reports"]["full_build_summary"], full_summary)
    _write_text(full_paths["reports"]["dataset_card"], _dataset_card_markdown(full_summary))
    _write_text(full_paths["reports"]["training_usage"], _training_usage_markdown(full_summary))
    return FullUtilityDatasetBuildResult(
        output_root=str(output),
        summary_path=str(full_paths["reports"]["full_build_summary"]),
        cost_summary_path=str(full_paths["reports"]["cost_summary"]),
        quality_report_path=str(full_paths["reports"]["quality_report"]),
        dataset_card_path=str(full_paths["reports"]["dataset_card"]),
        training_usage_path=str(full_paths["reports"]["training_usage"]),
        summary=full_summary,
    )


def build_expanded_utility_validator_dataset(
    *,
    repo_root: str | Path = ".",
    output_root: str | Path = "datasets/utility_validator",
    source: str = "all",
    backend: str = "dsapi",
    max_tasks: int = 160,
    include_text: bool = False,
    max_api_calls: int | None = None,
    confirm_cost_aware: bool = False,
    max_output_tokens: int = 128,
    timeout_seconds: float = 120.0,
    price_input_per_million: float | None = None,
    price_output_per_million: float | None = None,
    project_config_path: str | Path | None = None,
    model: str | None = None,
    batch_size: int = 50,
    checkpoint_api_calls: int = 100,
    max_estimated_cost_usd: float = 1.0,
    target_true_count: int = 150,
    target_false_count: int = 50,
    created_at: str | None = None,
) -> ExpandedUtilityDatasetBuildResult:
    if backend != "dsapi":
        raise ValueError("expanded_real requires --backend dsapi")
    if not confirm_cost_aware or max_api_calls is None or max_api_calls <= 0:
        raise ValueError("expanded_real requires --max-api-calls N --confirm-cost-aware")
    if include_text:
        raise ValueError("expanded_real refuses --include-text by default for prompt-safe training data")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if checkpoint_api_calls <= 0:
        raise ValueError("--checkpoint-api-calls must be positive")

    timestamp = created_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    root = Path(repo_root)
    output = Path(output_root)
    ensure_utility_dataset_layout(root)
    paths = _expanded_output_paths(output)
    _prepare_full_output_dirs(paths)
    checkpoint_root = output / "reports" / "expanded_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

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
    if backend_impl.model_name != "deepseek-v4-flash":
        raise ValueError(
            f"expanded_real requires exact model deepseek-v4-flash; effective model is {backend_impl.model_name!r}"
        )

    seed_rows = _load_existing_full_rows(output)
    seed_tasks = list(seed_rows["tasks"])
    seed_labels = list(seed_rows["labels"])
    seed_features = list(seed_rows["features"])
    seed_label_keys = {_label_dedupe_key(label) for label in seed_labels}

    source_result = build_utility_smoke_tasks(
        repo_root=root,
        source=source,
        max_tasks=None,
        include_text=False,
        created_at=timestamp,
    )
    base_tasks = _prepare_tasks_for_mode(source_result.tasks, source=source, mode="pilot_real", max_tasks=max_tasks)
    expanded_tasks = tuple(_mark_expanded_task(task) for task in base_tasks)
    batches = _batch_tasks(expanded_tasks, batch_size=batch_size)
    new_tasks: list[dict[str, Any]] = []
    new_labels: list[dict[str, Any]] = []
    new_features: list[dict[str, Any]] = []
    checkpoint_reports: list[dict[str, Any]] = []
    anomalous_batches: list[Mapping[str, Any]] = []
    total_cost_at_last_checkpoint = 0.0
    started_at = time.perf_counter()

    runner = UtilityLabelRunner(backend=backend_impl)
    for batch_index, batch_tasks in enumerate(batches, start=1):
        calls_before = backend_impl.calls_made
        cost_before = backend_impl.cost_summary()
        current_counts = _preserved_counts(seed_labels + new_labels)
        remaining_trainable = max(0, (target_true_count + target_false_count) - (current_counts["true"] + current_counts["false"]))
        remaining_false = max(0, target_false_count - current_counts["false"])
        rows = runner.run_candidates(
            tasks=batch_tasks,
            include_text=False,
            created_at=timestamp,
            strategies=EXPANDED_STRATEGIES,
            start_index=len(seed_labels) + len(new_labels) + 1,
            stop_after_trainable=max(remaining_trainable, 0) if remaining_trainable else None,
            target_false_count=remaining_false if remaining_false else None,
            max_api_call_delta=checkpoint_api_calls,
        )
        deduped_rows = tuple(row for row in rows if _label_dedupe_key(row["label"]) not in seed_label_keys)
        for row in deduped_rows:
            seed_label_keys.add(_label_dedupe_key(row["label"]))
        labels = tuple(row["label"] for row in deduped_rows)
        features = tuple(row["feature"] for row in deduped_rows if row["label"].get("is_utility_preserved") in {True, False})
        tasks = tuple(_strip_private_task_fields(row["task"]) for row in deduped_rows)
        cost_after = backend_impl.cost_summary()
        checkpoint_report = _checkpoint_report(
            batch_index=batch_index,
            tasks=tasks,
            labels=labels,
            features=features,
            cost_before=cost_before,
            cost_after=cost_after,
            max_estimated_cost_usd=max_estimated_cost_usd,
            max_api_calls=max_api_calls,
            checkpoint_api_calls=checkpoint_api_calls,
            output_root=output,
            created_at=timestamp,
            scan_splits=EXPANDED_SPLITS,
            report_names=_expanded_report_file_names(),
        )
        batch_dir = checkpoint_root / f"batch_{batch_index:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(batch_dir / "utility_tasks.jsonl", tasks)
        _write_jsonl(batch_dir / "utility_labels.jsonl", labels)
        _write_jsonl(batch_dir / "training_features.jsonl", features)
        _write_json(batch_dir / "checkpoint_report.json", checkpoint_report)
        checkpoint_reports.append(checkpoint_report)
        if not checkpoint_report["passed"]:
            anomalous_batches.append(
                {
                    "batch_index": batch_index,
                    "report_path": str(batch_dir / "checkpoint_report.json"),
                    "blockers": checkpoint_report["blockers"],
                    "handled": "paused_before_merge",
                }
            )
            break
        new_tasks.extend(dict(task) for task in tasks)
        new_labels.extend(dict(label) for label in labels)
        new_features.extend(dict(feature) for feature in features)
        total_cost_at_last_checkpoint = float(cost_after.get("estimated_cost_usd") or 0.0)
        merged_counts = _preserved_counts(seed_labels + new_labels)
        if (merged_counts["true"] + merged_counts["false"]) >= target_true_count + target_false_count and merged_counts["false"] >= target_false_count:
            break
        if backend_impl.calls_made >= max_api_calls:
            break
        if backend_impl.calls_made == calls_before and rows:
            raise RuntimeError("expanded checkpoint made no API-call progress despite emitted rows")

    raw_new_label_count = len(new_labels)
    raw_new_feature_count = len(new_features)
    raw_new_task_count = len(new_tasks)
    selected_rows = _select_expanded_rows(
        tasks=seed_tasks + new_tasks,
        labels=seed_labels + new_labels,
        features=seed_features + new_features,
        target_true_count=target_true_count,
        target_false_count=target_false_count,
        seed_label_ids={str(label.get("label_id") or "") for label in seed_labels},
        seed_feature_label_ids={str(feature.get("label_id") or "") for feature in seed_features},
    )
    all_tasks = list(selected_rows["tasks"])
    all_labels = list(selected_rows["labels"])
    all_features = list(selected_rows["features"])
    selection_report = selected_rows["selection_report"]
    split_rows = _split_full_rows(all_tasks, all_labels, all_features, split_names=EXPANDED_SPLITS)
    _write_split_outputs(paths, split_rows)
    cost_summary = _full_cost_summary(backend_impl.cost_summary())
    quality_report = _full_quality_report(
        split_rows=split_rows,
        checkpoint_reports=checkpoint_reports,
        anomalous_batches=anomalous_batches,
        output_root=output,
        created_at=timestamp,
        split_names=EXPANDED_SPLITS,
        scan_report_names=_expanded_report_file_names(),
        schema_version="utility-validator-expanded-quality-report-v1",
    )
    elapsed_seconds = time.perf_counter() - started_at
    final_counts = _preserved_counts(all_labels)
    extra_fields = {
        "seed_label_count": len(seed_labels),
        "seed_feature_count": len(seed_features),
        "raw_new_task_count": raw_new_task_count,
        "raw_new_label_count": raw_new_label_count,
        "raw_new_feature_count": raw_new_feature_count,
        "new_label_count": selection_report.get("selected_new_label_count"),
        "new_feature_count": selection_report.get("selected_new_feature_count"),
        "target_selection_report": selection_report,
        "target_true_count": target_true_count,
        "target_false_count": target_false_count,
        "target_trainable_count": target_true_count + target_false_count,
        "target_reached": (
            final_counts["true"] >= target_true_count
            and final_counts["false"] >= target_false_count
            and final_counts["true"] + final_counts["false"] >= target_true_count + target_false_count
        ),
        "candidate_strategies": EXPANDED_STRATEGIES,
        "candidate_strategy_false_counts": quality_report.get("candidate_strategy_false_counts"),
    }
    summary = _full_build_summary(
        source=source,
        max_tasks=max_tasks,
        batch_size=batch_size,
        checkpoint_api_calls=checkpoint_api_calls,
        max_api_calls=max_api_calls,
        include_text=include_text,
        created_at=timestamp,
        elapsed_seconds=elapsed_seconds,
        split_rows=split_rows,
        cost_summary=cost_summary,
        quality_report=quality_report,
        checkpoint_reports=checkpoint_reports,
        anomalous_batches=anomalous_batches,
        output_paths=paths,
        source_reports=source_result.source_reports,
        total_cost_at_last_checkpoint=total_cost_at_last_checkpoint,
        split_names=EXPANDED_SPLITS,
        mode="expanded_real",
        schema_version="utility-validator-expanded-build-summary-v1",
        extra_fields=extra_fields,
        exception_policy="anomalous batches are written under reports/expanded_checkpoints and not merged",
    )
    _write_json(paths["reports"]["cost_summary"], cost_summary)
    _write_json(paths["reports"]["quality_report"], quality_report)
    _write_json(paths["reports"]["full_build_summary"], summary)
    _write_text(paths["reports"]["dataset_card"], _dataset_card_markdown(summary))
    _write_text(paths["reports"]["training_usage"], _training_usage_markdown(summary))
    return ExpandedUtilityDatasetBuildResult(
        output_root=str(output),
        summary_path=str(paths["reports"]["full_build_summary"]),
        cost_summary_path=str(paths["reports"]["cost_summary"]),
        quality_report_path=str(paths["reports"]["quality_report"]),
        dataset_card_path=str(paths["reports"]["dataset_card"]),
        training_usage_path=str(paths["reports"]["training_usage"]),
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

    def run_candidates(
        self,
        *,
        tasks: Sequence[Mapping[str, Any]],
        include_text: bool,
        created_at: str,
        strategies: Sequence[str] = EXPANDED_STRATEGIES,
        start_index: int = 1,
        stop_after_trainable: int | None = None,
        target_false_count: int | None = None,
        max_api_call_delta: int | None = None,
    ) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        trainable_count = 0
        false_count = 0
        calls_start = self.backend.calls_made if self.backend.name == "dsapi" else 0
        for task in tasks:
            if self.backend.name == "dsapi" and self.backend.remaining_calls() <= 0:
                break
            if (
                self.backend.name == "dsapi"
                and max_api_call_delta is not None
                and self.backend.calls_made - calls_start >= max_api_call_delta
            ):
                break
            for row in self._run_task_candidates(
                task=task,
                include_text=include_text,
                created_at=created_at,
                start_index=start_index + len(rows),
                strategies=strategies,
                calls_start=calls_start,
                max_api_call_delta=max_api_call_delta,
            ):
                rows.append(row)
                if row["label"].get("is_utility_preserved") in {True, False}:
                    trainable_count += 1
                if row["label"].get("is_utility_preserved") is False:
                    false_count += 1
                if stop_after_trainable is not None and trainable_count >= stop_after_trainable:
                    if target_false_count is None or false_count >= target_false_count:
                        return tuple(rows)
                if self.backend.name == "dsapi" and self.backend.remaining_calls() <= 0:
                    return tuple(rows)
                if (
                    self.backend.name == "dsapi"
                    and max_api_call_delta is not None
                    and self.backend.calls_made - calls_start >= max_api_call_delta
                ):
                    return tuple(rows)
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
        return self._run_candidate_label(
            task=task,
            include_text=include_text,
            created_at=created_at,
            index=index,
            compile_result=compile_result,
            plan=plan,
            original_output=None,
            original_trace=None,
            original_result=None,
        )

    def _run_task_candidates(
        self,
        *,
        task: Mapping[str, Any],
        include_text: bool,
        created_at: str,
        start_index: int,
        strategies: Sequence[str],
        calls_start: int = 0,
        max_api_call_delta: int | None = None,
    ) -> tuple[dict[str, Any], ...]:
        messages = _messages_for_task(task)
        session_id = f"utility-dataset-expanded:{task.get('dataset_source')}"
        compile_result = self.compiler.compile(messages, session_id=session_id)
        candidates = _expanded_candidates_for_task(
            planner=self.planner,
            compile_result=compile_result,
            session_id=session_id,
            strategies=strategies,
        )
        rows: list[dict[str, Any]] = []
        original_output: Any = None
        original_trace: Any = None
        original_result: OracleResult | None = None
        for offset, candidate in enumerate(candidates):
            if self.backend.name == "dsapi" and self.backend.remaining_calls() <= 0:
                break
            if (
                self.backend.name == "dsapi"
                and max_api_call_delta is not None
                and self.backend.calls_made - calls_start >= max_api_call_delta
            ):
                break
            if (
                self.backend.name == "dsapi"
                and max_api_call_delta is not None
                and original_result is None
                and self.backend.calls_made - calls_start >= max_api_call_delta - 1
            ):
                break
            if (
                self.backend.name == "dsapi"
                and max_api_call_delta is not None
                and original_result is not None
                and original_result.passed
                and self.backend.calls_made - calls_start >= max_api_call_delta
            ):
                break
            plan = self.planner._plan_from_candidate(compile_result, candidate)
            row = self._run_candidate_label(
                task=task,
                include_text=include_text,
                created_at=created_at,
                index=start_index + offset,
                compile_result=compile_result,
                plan=plan,
                original_output=original_output,
                original_trace=original_trace,
                original_result=original_result,
            )
            rows.append(row)
            original_run = row.get("_local_original_run")
            if isinstance(original_run, Mapping) and original_result is None:
                original_output = original_run.get("output")
                original_trace = original_run.get("trace")
                maybe_result = original_run.get("result")
                original_result = maybe_result if isinstance(maybe_result, OracleResult) else None
            if original_result is not None and not original_result.passed:
                break
        self.planner.observe(compile_result, session_id=session_id)
        return tuple(rows)

    def _run_candidate_label(
        self,
        *,
        task: Mapping[str, Any],
        include_text: bool,
        created_at: str,
        index: int,
        compile_result: Any,
        plan: Any,
        original_output: Any,
        original_trace: Any,
        original_result: OracleResult | None,
    ) -> dict[str, Any]:
        rewritten_messages = rewrite_messages(compile_result, plan)
        validation = self.validator.validate(
            original_messages=compile_result.messages,
            rewritten_messages=rewritten_messages,
            compile_result=compile_result,
            plan=plan,
        )
        candidate = plan.prefix_tree_candidate
        agent_id = (
            next(iter(candidate.agent_block_orders.keys()))
            if candidate and candidate.agent_block_orders
            else str(getattr(compile_result, "session_id", "") or task.get("task_id") or "agent")
        )
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
        if original_result is None:
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
            "_local_original_run": {
                "output": original_output,
                "trace": original_trace,
                "result": original_result,
            },
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
        "datasets/utility_validator/tasks/expanded_train",
        "datasets/utility_validator/tasks/expanded_valid",
        "datasets/utility_validator/tasks/expanded_test",
        "datasets/utility_validator/labels/smoke",
        "datasets/utility_validator/labels/smoke_real",
        "datasets/utility_validator/labels/pilot_real",
        "datasets/utility_validator/labels/train",
        "datasets/utility_validator/labels/valid",
        "datasets/utility_validator/labels/test",
        "datasets/utility_validator/labels/expanded_train",
        "datasets/utility_validator/labels/expanded_valid",
        "datasets/utility_validator/labels/expanded_test",
        "datasets/utility_validator/features/smoke",
        "datasets/utility_validator/features/smoke_real",
        "datasets/utility_validator/features/pilot_real",
        "datasets/utility_validator/features/train",
        "datasets/utility_validator/features/valid",
        "datasets/utility_validator/features/test",
        "datasets/utility_validator/features/expanded_train",
        "datasets/utility_validator/features/expanded_valid",
        "datasets/utility_validator/features/expanded_test",
        "datasets/utility_validator/reports",
        "datasets/utility_validator/reports/full_checkpoints",
        "datasets/utility_validator/reports/expanded_checkpoints",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build first-stage Utility Validator smoke dataset artifacts.")
    parser.add_argument("--repo-root", default=".", help="Repository root. Defaults to current working directory.")
    parser.add_argument("--output-root", default="datasets/utility_validator", help="Utility validator dataset root.")
    parser.add_argument("--output-dir", help="Alias for --output-root for compatibility with earlier task notes.")
    parser.add_argument("--source", default="all", help="all or comma-separated source names.")
    parser.add_argument("--mode", default="smoke", choices=("smoke", "smoke_real", "pilot_real", "train_real", "expanded_real"))
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
    parser.add_argument("--batch-size", type=int, default=25, help="train_real checkpoint label batch size.")
    parser.add_argument("--checkpoint-api-calls", type=int, default=100, help="train_real max calls per checkpoint.")
    parser.add_argument("--max-estimated-cost-usd", type=float, default=1.0, help="train_real pause budget.")
    parser.add_argument("--target-true-count", type=int, default=150, help="expanded_real target preserved=true count.")
    parser.add_argument("--target-false-count", type=int, default=50, help="expanded_real target preserved=false count.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_dir or args.output_root
    if args.mode == "expanded_real":
        result = build_expanded_utility_validator_dataset(
            repo_root=args.repo_root,
            output_root=output_root,
            source=args.source,
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
            batch_size=args.batch_size,
            checkpoint_api_calls=args.checkpoint_api_calls,
            max_estimated_cost_usd=args.max_estimated_cost_usd,
            target_true_count=args.target_true_count,
            target_false_count=args.target_false_count,
        )
    elif args.mode == "train_real":
        result = build_full_utility_validator_dataset(
            repo_root=args.repo_root,
            output_root=output_root,
            source=args.source,
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
            batch_size=args.batch_size,
            checkpoint_api_calls=args.checkpoint_api_calls,
            max_estimated_cost_usd=args.max_estimated_cost_usd,
        )
    else:
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


def _expanded_candidates_for_task(
    *,
    planner: HierarchicalPrefixPlanner,
    compile_result: Any,
    session_id: str,
    strategies: Sequence[str],
) -> tuple[Any, ...]:
    wanted = tuple(dict.fromkeys(str(strategy) for strategy in strategies))
    candidates = list(planner.generate_candidates(compile_result, session_id=session_id, top_k=8))
    if "adversarial" in wanted:
        adversarial = _adversarial_candidate_for_task(
            planner=planner,
            compile_result=compile_result,
            session_id=session_id,
        )
        if adversarial is not None:
            candidates.append(adversarial)
    by_strategy: dict[str, Any] = {}
    for candidate in candidates:
        strategy = _candidate_strategy(candidate)
        by_strategy.setdefault(strategy, candidate)
    selected = [by_strategy[strategy] for strategy in wanted if strategy in by_strategy]
    if not selected and candidates:
        selected.append(candidates[0])
    return tuple(selected)


def _adversarial_candidate_for_task(
    *,
    planner: HierarchicalPrefixPlanner,
    compile_result: Any,
    session_id: str,
) -> Any | None:
    blocks = tuple(getattr(compile_result, "blocks", ()) or ())
    if not blocks:
        return None
    promoted = tuple(_adversarial_promoted_blocks(blocks))
    if not promoted:
        return None
    return planner._build_candidate(
        compile_result=compile_result,
        session_id=session_id,
        policy_name="adversarial",
        promoted_blocks=promoted,
        generation_reason="boundary candidate moves conditional format/tool/state context while hard gate still blocks local-only risks",
    )


def _adversarial_promoted_blocks(blocks: Sequence[Any]) -> tuple[Any, ...]:
    preferred_semantics = {
        SemanticType.OUTPUT_FORMAT,
        SemanticType.SHARED_TOOL_DESCRIPTION,
        SemanticType.SHARED_CONTEXT,
        SemanticType.TEAM_POLICY,
        SemanticType.GLOBAL_TASK_BACKGROUND,
    }
    candidates = [
        block
        for block in blocks
        if getattr(block, "is_system_text", False)
        and getattr(block, "movability", None) in {Movability.SAFE_PREFIX, Movability.CONDITIONAL_PREFIX}
        and getattr(block, "share_scope", None) in {ShareScope.GLOBAL, ShareScope.SUBGROUP}
        and getattr(block, "semantic_type", None) in preferred_semantics
        and not _block_has_nonshareable_training_risk(block)
    ]
    if not candidates:
        return ()
    ordered = sorted(
        candidates,
        key=lambda block: (
            {
                SemanticType.OUTPUT_FORMAT: 0,
                SemanticType.SHARED_TOOL_DESCRIPTION: 1,
                SemanticType.SHARED_CONTEXT: 2,
                SemanticType.TEAM_POLICY: 3,
                SemanticType.GLOBAL_TASK_BACKGROUND: 4,
            }.get(getattr(block, "semantic_type", None), 9),
            -int(getattr(block, "original_position").message_index),
            -int(getattr(block, "original_position").part_index),
            str(getattr(block, "block_id", "")),
        ),
    )
    return tuple(ordered[:3])


def _block_has_nonshareable_training_risk(block: Any) -> bool:
    return bool(
        getattr(block, "contains_private_info", False)
        or getattr(block, "contains_role_identity", False)
        or getattr(block, "contains_tool_permission", False)
        or getattr(block, "contains_latest_user_instruction", False)
        or getattr(block, "contains_tool_result", False)
        or getattr(block, "contains_credential", False)
    )


def _output_paths(output: Path, mode: str) -> dict[str, Path]:
    summary_name = "real_smoke_summary.json" if mode == "smoke_real" else f"{mode}_summary.json"
    return {
        "tasks": output / "tasks" / mode / "utility_tasks.jsonl",
        "labels": output / "labels" / mode / "utility_labels.jsonl",
        "features": output / "features" / mode / "training_features.jsonl",
        "summary": output / "reports" / summary_name,
    }


def _full_output_paths(output: Path) -> dict[str, Any]:
    return {
        "tasks": {split: output / "tasks" / split / "utility_tasks.jsonl" for split in FINAL_SPLITS},
        "labels": {split: output / "labels" / split / "utility_labels.jsonl" for split in FINAL_SPLITS},
        "features": {split: output / "features" / split / "training_features.jsonl" for split in FINAL_SPLITS},
        "feature_aliases": {
            split: output / "features" / split / f"{split}_features.jsonl" for split in FINAL_SPLITS
        },
        "reports": {
            "full_build_summary": output / "reports" / "full_build_summary.json",
            "cost_summary": output / "reports" / "cost_summary.json",
            "quality_report": output / "reports" / "quality_report.json",
            "dataset_card": output / "reports" / "dataset_card.md",
            "training_usage": output / "reports" / "training_usage.md",
        },
    }


def _expanded_output_paths(output: Path) -> dict[str, Any]:
    return {
        "tasks": {split: output / "tasks" / split / "utility_tasks.jsonl" for split in EXPANDED_SPLITS},
        "labels": {split: output / "labels" / split / "utility_labels.jsonl" for split in EXPANDED_SPLITS},
        "features": {split: output / "features" / split / "training_features.jsonl" for split in EXPANDED_SPLITS},
        "feature_aliases": {
            "expanded_train": output / "features" / "expanded_train" / "train_features.jsonl",
            "expanded_valid": output / "features" / "expanded_valid" / "valid_features.jsonl",
            "expanded_test": output / "features" / "expanded_test" / "test_features.jsonl",
        },
        "reports": {
            "full_build_summary": output / "reports" / "expanded_build_summary.json",
            "cost_summary": output / "reports" / "expanded_cost_summary.json",
            "quality_report": output / "reports" / "expanded_quality_report.json",
            "dataset_card": output / "reports" / "expanded_dataset_card.md",
            "training_usage": output / "reports" / "expanded_training_usage.md",
        },
    }


def _prepare_full_output_dirs(paths: Mapping[str, Any]) -> None:
    for group in ("tasks", "labels", "features", "feature_aliases", "reports"):
        values = paths.get(group)
        if isinstance(values, Mapping):
            for path in values.values():
                Path(path).parent.mkdir(parents=True, exist_ok=True)
    split_names = tuple(paths.get("tasks", {}).keys()) if isinstance(paths.get("tasks"), Mapping) else FINAL_SPLITS
    for split in split_names:
        for group in ("tasks", "labels", "features", "feature_aliases"):
            if split not in paths.get(group, {}):
                continue
            path = paths[group][split]
            if Path(path).exists():
                Path(path).unlink()


def _select_expanded_rows(
    *,
    tasks: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    target_true_count: int,
    target_false_count: int,
    seed_label_ids: set[str] | None = None,
    seed_feature_label_ids: set[str] | None = None,
) -> Mapping[str, Any]:
    seed_label_ids = seed_label_ids or set()
    seed_feature_label_ids = seed_feature_label_ids or set()
    unique_labels = _unique_labels_by_id(labels)
    true_labels = tuple(label for label in unique_labels if label.get("is_utility_preserved") is True)
    false_labels = tuple(label for label in unique_labels if label.get("is_utility_preserved") is False)
    skipped_labels = tuple(label for label in unique_labels if label.get("is_utility_preserved") not in {True, False})
    selected_false = _stratified_label_take(false_labels, target_false_count)
    selected_true = _stratified_label_take(true_labels, target_true_count)
    selected_trainable_ids = {
        str(label.get("label_id"))
        for label in (*selected_true, *selected_false)
        if label.get("label_id") is not None
    }
    selected_skipped = tuple(skipped_labels)
    selected_labels = _sort_labels_for_output((*selected_true, *selected_false, *selected_skipped))
    feature_by_label = _feature_by_label_id(features)
    selected_features = tuple(
        _feature_with_label_metadata(feature_by_label[str(label.get("label_id"))], label)
        for label in selected_labels
        if str(label.get("label_id")) in selected_trainable_ids
        and str(label.get("label_id")) in feature_by_label
    )
    task_by_id = _task_by_id(tasks)
    selected_task_ids = []
    seen_task_ids: set[str] = set()
    for label in selected_labels:
        task_id = str(label.get("task_id") or "")
        if not task_id or task_id in seen_task_ids:
            continue
        seen_task_ids.add(task_id)
        selected_task_ids.append(task_id)
    selected_tasks = tuple(
        task_by_id.get(task_id) or _task_stub_from_label(next(label for label in selected_labels if str(label.get("task_id") or "") == task_id))
        for task_id in selected_task_ids
    )
    feature_counts = _preserved_counts(selected_features)
    label_counts = _preserved_counts(selected_labels)
    return {
        "tasks": selected_tasks,
        "labels": selected_labels,
        "features": selected_features,
        "selection_report": {
            "policy": (
                "keep real false labels up to target_false_count, select true labels by deterministic "
                "source/strategy strata up to target_true_count, keep original-failed labels as skipped audit rows, "
                "and exclude skipped rows from features"
            ),
            "raw_label_count": len(unique_labels),
            "raw_feature_count": len(features),
            "raw_is_utility_preserved_counts": _preserved_counts(unique_labels),
            "selected_task_count": len(selected_tasks),
            "selected_label_count": len(selected_labels),
            "selected_feature_count": len(selected_features),
            "selected_is_utility_preserved_counts": label_counts,
            "selected_feature_is_utility_preserved_counts": feature_counts,
            "selected_new_label_count": sum(
                1 for label in selected_labels if str(label.get("label_id") or "") not in seed_label_ids
            ),
            "selected_new_feature_count": sum(
                1 for feature in selected_features if str(feature.get("label_id") or "") not in seed_feature_label_ids
            ),
            "selected_false_count": label_counts["false"],
            "selected_true_count": label_counts["true"],
            "selected_skipped_count": label_counts["skipped"],
            "target_false_count": target_false_count,
            "target_true_count": target_true_count,
        },
    }


def _unique_labels_by_id(labels: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    seen: set[str] = set()
    unique: list[Mapping[str, Any]] = []
    for label in labels:
        label_id = str(label.get("label_id") or "")
        if not label_id:
            label_id = f"{label.get('task_id')}::{label.get('candidate_strategy')}::{len(unique)}"
        if label_id in seen:
            continue
        seen.add(label_id)
        unique.append(label)
    return tuple(unique)


def _feature_by_label_id(features: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    feature_by_label: dict[str, Mapping[str, Any]] = {}
    for feature in features:
        label_id = str(feature.get("label_id") or "")
        if label_id and label_id not in feature_by_label:
            feature_by_label[label_id] = feature
    return feature_by_label


def _feature_with_label_metadata(feature: Mapping[str, Any], label: Mapping[str, Any]) -> Mapping[str, Any]:
    enriched = dict(feature)
    enriched.setdefault("original_run_status", label.get("original_run_status"))
    enriched.setdefault("reordered_run_status", label.get("reordered_run_status"))
    enriched.setdefault("api_model", label.get("api_model"))
    enriched.setdefault("input_tokens", label.get("input_tokens"))
    enriched.setdefault("output_tokens", label.get("output_tokens"))
    enriched.setdefault("cached_tokens", label.get("cached_tokens"))
    enriched.setdefault("latency", label.get("latency"))
    enriched.setdefault("estimated_cost", label.get("estimated_cost"))
    return enriched


def _task_by_id(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    task_by_id: dict[str, Mapping[str, Any]] = {}
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        if task_id and task_id not in task_by_id:
            task_by_id[task_id] = task
    return task_by_id


def _unique_tasks_by_id(tasks: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    seen: set[str] = set()
    unique: list[Mapping[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        if task_id and task_id in seen:
            continue
        if task_id:
            seen.add(task_id)
        unique.append(task)
    return tuple(unique)


def _task_stub_from_label(label: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "schema_version": "utility-task-v1",
        "task_id": label.get("task_id"),
        "dataset_source": label.get("dataset_source"),
        "scenario_type": label.get("scenario_type"),
        "prompt_text_included": False,
        "source_metadata": {"recovered_from_label": True},
    }


def _stratified_label_take(labels: Sequence[Mapping[str, Any]], target: int) -> tuple[Mapping[str, Any], ...]:
    unique = _sort_labels_for_output(_unique_labels_by_id(labels))
    if target <= 0 or not unique:
        return ()
    if len(unique) <= target:
        return tuple(unique)
    selected: list[Mapping[str, Any]] = []
    selected_ids: set[str] = set()
    base_quota = target // len(DATASET_SOURCES)
    remainder = target % len(DATASET_SOURCES)
    for source_index, source in enumerate(DATASET_SOURCES):
        quota = base_quota + (1 if source_index < remainder else 0)
        source_labels = tuple(label for label in unique if str(label.get("dataset_source") or "") == source)
        for label in _round_robin_labels_by_strategy(source_labels, quota):
            label_id = str(label.get("label_id") or "")
            if label_id and label_id not in selected_ids:
                selected.append(label)
                selected_ids.add(label_id)
    for label in unique:
        if len(selected) >= target:
            break
        label_id = str(label.get("label_id") or "")
        if label_id and label_id not in selected_ids:
            selected.append(label)
            selected_ids.add(label_id)
    return tuple(selected)


def _round_robin_labels_by_strategy(labels: Sequence[Mapping[str, Any]], quota: int) -> tuple[Mapping[str, Any], ...]:
    if quota <= 0:
        return ()
    sorted_labels = _sort_labels_for_output(labels)
    strategy_order = (*EXPANDED_STRATEGIES, "none", "unknown")
    buckets: dict[str, list[Mapping[str, Any]]] = {strategy: [] for strategy in strategy_order}
    for label in sorted_labels:
        buckets.setdefault(str(label.get("candidate_strategy") or "unknown"), []).append(label)
    selected: list[Mapping[str, Any]] = []
    while len(selected) < quota:
        progressed = False
        for strategy in strategy_order:
            bucket = buckets.get(strategy) or []
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) >= quota:
                break
        if not progressed:
            break
    return tuple(selected)


def _sort_labels_for_output(labels: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    source_rank = {source: index for index, source in enumerate(DATASET_SOURCES)}
    strategy_rank = {strategy: index for index, strategy in enumerate(EXPANDED_STRATEGIES)}
    strategy_rank.setdefault("none", len(strategy_rank))
    strategy_rank.setdefault("unknown", len(strategy_rank))
    return tuple(
        sorted(
            labels,
            key=lambda label: (
                source_rank.get(str(label.get("dataset_source") or ""), len(source_rank)),
                str(label.get("task_id") or ""),
                strategy_rank.get(str(label.get("candidate_strategy") or "unknown"), len(strategy_rank)),
                str(label.get("label_id") or ""),
            ),
        )
    )


def _batch_tasks(tasks: Sequence[Mapping[str, Any]], *, batch_size: int) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    return tuple(tuple(tasks[index : index + batch_size]) for index in range(0, len(tasks), batch_size))


def _checkpoint_report(
    *,
    batch_index: int,
    tasks: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    cost_before: Mapping[str, Any],
    cost_after: Mapping[str, Any],
    max_estimated_cost_usd: float,
    max_api_calls: int,
    checkpoint_api_calls: int,
    output_root: Path,
    created_at: str,
    scan_splits: Sequence[str] = FINAL_SPLITS,
    report_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    calls_delta = _int(cost_after.get("api_call_count")) - _int(cost_before.get("api_call_count"))
    cost_delta = float(cost_after.get("estimated_cost_usd") or 0.0) - float(cost_before.get("estimated_cost_usd") or 0.0)
    latency_delta = float(cost_after.get("latency_seconds_total") or 0.0) - float(cost_before.get("latency_seconds_total") or 0.0)
    label_source_counts = Counter(str(label.get("label_source")) for label in labels)
    preserved_counts = Counter(str(label.get("is_utility_preserved")) for label in labels)
    failure_counts = Counter(str(label.get("failure_type") or "none") for label in labels)
    source_counts = Counter(str(label.get("dataset_source")) for label in labels)
    trainable_label_count = sum(1 for label in labels if label.get("is_utility_preserved") in {True, False})
    runtime_error_count = failure_counts.get("runtime_error", 0)
    original_failed_count = sum(1 for label in labels if label.get("original_run_status") == "failed")
    schema_like_errors = _schema_parse_error_count(labels)
    prompt_text_saved = _has_prompt_text(labels) or _has_prompt_text(features)
    scan_report = _scan_prompt_safe_outputs(output_root, split_names=scan_splits, report_names=report_names)
    jsonl_report = _jsonl_readability_report(output_root)
    checks = {
        "model_is_deepseek_v4_flash": cost_after.get("model") == "deepseek-v4-flash",
        "model_is_not_pro": not bool(cost_after.get("model_is_pro")),
        "label_source_all_dsapi": bool(labels)
        and all(label.get("label_source") == "dsapi_execution_oracle" for label in labels),
        "no_fake_smoke_oracle": all(label.get("label_source") != "fake_smoke_oracle" for label in labels),
        "prompt_text_saved_false": not prompt_text_saved,
        "no_key_prompt_or_private_scan_hits": not scan_report["has_hits"],
        "original_failed_skipped_not_in_features": _original_failed_skipped_not_in_features(labels, features),
        "has_true_false_skipped_counts": bool(labels) and any(key in preserved_counts for key in ("True", "False", "None")),
        "failure_type_distribution_ok": bool(failure_counts),
        "source_coverage_ok": all(source in source_counts for source in DATASET_SOURCES) if batch_index == 1 else bool(source_counts),
        "cost_within_budget": float(cost_after.get("estimated_cost_usd") or 0.0) <= max_estimated_cost_usd,
        "latency_not_abnormal": (latency_delta / calls_delta if calls_delta else 0.0) <= 8.0,
        "runtime_schema_parse_error_rate_ok": (
            (runtime_error_count + schema_like_errors) / len(labels) <= 0.2 if labels else False
        ),
        "feature_count_matches_trainable_labels": len(features) == trainable_label_count,
        "jsonl_schema_readable": jsonl_report["readable"],
        "checkpoint_api_calls_within_limit": calls_delta <= checkpoint_api_calls,
        "total_api_calls_within_limit": _int(cost_after.get("api_call_count")) <= max_api_calls,
        "original_failed_rate_ok": (original_failed_count / len(labels) <= 0.4 if labels else False),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "utility-validator-full-checkpoint-v1",
        "created_at": created_at,
        "batch_index": batch_index,
        "passed": not blockers,
        "blockers": blockers,
        "checks": checks,
        "task_count": len(tasks),
        "label_count": len(labels),
        "feature_count": len(features),
        "trainable_label_count": trainable_label_count,
        "label_source_counts": dict(sorted(label_source_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "is_utility_preserved_counts": dict(sorted(preserved_counts.items())),
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "runtime_error_rate": runtime_error_count / len(labels) if labels else 0.0,
        "original_failed_rate": original_failed_count / len(labels) if labels else 0.0,
        "api_call_count_delta": calls_delta,
        "estimated_cost_delta": cost_delta,
        "latency_seconds_avg_delta": latency_delta / calls_delta if calls_delta else 0.0,
        "cost_summary_after": {
            "api_call_count": cost_after.get("api_call_count"),
            "model": cost_after.get("model"),
            "model_is_v4_flash": cost_after.get("model_is_v4_flash"),
            "model_is_pro": cost_after.get("model_is_pro"),
            "total_input_tokens": cost_after.get("total_input_tokens"),
            "total_output_tokens": cost_after.get("total_output_tokens"),
            "total_cached_tokens": cost_after.get("total_cached_tokens"),
            "estimated_cost_usd": cost_after.get("estimated_cost_usd"),
            "latency_seconds_avg": cost_after.get("latency_seconds_avg"),
        },
        "scan_report": scan_report,
        "jsonl_report": jsonl_report,
        "exception_handling": "merged_only_if_passed",
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
        "sample_id": _safe_id(f"uv_feature_{sample_index:04d}_{task.get('task_id')}_{label.get('candidate_strategy')}"),
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
        "original_run_status": label.get("original_run_status"),
        "reordered_run_status": label.get("reordered_run_status"),
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
    if task.get("_local_dsapi_prompt_sensitive") is True:
        return _dsapi_prompt_sensitive_system_instruction(oracle_type=oracle_type, oracle_spec=oracle_spec)
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
    if task.get("_local_dsapi_prompt_sensitive") is True:
        return "\n\n".join(
            [
                _dsapi_prompt_sensitive_output_shape(task),
                "PROMPT_UNDER_TEST_START",
                prompt,
                "PROMPT_UNDER_TEST_END",
                "Execute the prompt under test as written. Return only the JSON object.",
            ]
        )
    return "\n\n".join(
        [
            _dsapi_output_contract(task),
            "PROMPT_UNDER_TEST_START",
            prompt,
            "PROMPT_UNDER_TEST_END",
        ]
    )


def _dsapi_prompt_sensitive_system_instruction(*, oracle_type: str, oracle_spec: Mapping[str, Any]) -> str:
    common = (
        "You are executing a candidate prompt for Utility Validator labeling. "
        "Treat the prompt under test as the task instruction. The harness only specifies the JSON response shape. "
        "If the prompt under test has lost, weakened, or reordered an important constraint, follow the prompt under test "
        "rather than correcting it from the harness. Return exactly one JSON object and no prose. "
    )
    if oracle_type == "unit_test":
        return common + (
            "Return fields status and tests_passed. Set tests_passed true only if the prompt under test preserves "
            "the code task, function signature, boundary conditions, and test intent; otherwise set false."
        )
    if oracle_type == "json_schema":
        return common + (
            "Return the JSON object requested by the prompt under test. Do not add fields unless the prompt under test "
            "requests them. Do not repair missing or weakened format constraints using hidden oracle knowledge."
        )
    if oracle_type == "tool_trace":
        return common + (
            "Return fields status, tool_calls, and final_result. Choose the tool calls implied by the prompt under test. "
            "If role/tool permission placement makes the allowed tool ambiguous, reflect that ambiguity in the chosen tool_calls."
        )
    if oracle_type == "privacy_check":
        return common + (
            "Return fields status, text, and leaked_private_memory. Write the response requested by the prompt under test. "
            "Set leaked_private_memory true if the response reveals private memory or secret-like content."
        )
    if oracle_type == "role_check":
        return common + (
            "Return fields status, text, and role_boundary_ok. Respond according to the prompt under test. "
            "Set role_boundary_ok false if the response speaks as the wrong role or claims another role's authority."
        )
    if oracle_type == "state_check":
        return common + (
            "Return fields status, state_consistent, and observed_event_order. Use the event order implied by the prompt under test; "
            "set state_consistent false if the state/order constraint is lost or contradicted."
        )
    return common + "Return a JSON object describing the prompt execution."


def _dsapi_prompt_sensitive_output_shape(task: Mapping[str, Any]) -> str:
    oracle_type = str(task.get("expected_oracle_type") or "")
    oracle_spec = (
        task.get("_local_oracle_spec")
        if isinstance(task.get("_local_oracle_spec"), Mapping)
        else task.get("oracle_spec")
        if isinstance(task.get("oracle_spec"), Mapping)
        else {}
    )
    if oracle_type == "unit_test":
        return 'OUTPUT_SHAPE: {"status": "passed|failed", "tests_passed": boolean}.'
    if oracle_type == "json_schema":
        schema = oracle_spec.get("json_schema") if isinstance(oracle_spec.get("json_schema"), Mapping) else {}
        required = tuple(str(field) for field in schema.get("required") or ())
        properties = tuple(str(field) for field in (schema.get("properties") or {}).keys()) if isinstance(schema.get("properties"), Mapping) else ()
        return (
            "OUTPUT_SHAPE: Return one JSON object. Oracle expected fields are "
            + json.dumps({"required": required, "properties": properties}, ensure_ascii=False)
            + ", but obey the prompt under test if it conflicts."
        )
    if oracle_type == "tool_trace":
        return 'OUTPUT_SHAPE: {"status": "passed|failed", "tool_calls": [{"name": string}], "final_result": string}.'
    if oracle_type == "privacy_check":
        return 'OUTPUT_SHAPE: {"status": "passed|failed", "text": string, "leaked_private_memory": boolean}.'
    if oracle_type == "role_check":
        return 'OUTPUT_SHAPE: {"status": "passed|failed", "text": string, "role_boundary_ok": boolean}.'
    if oracle_type == "state_check":
        return 'OUTPUT_SHAPE: {"status": "passed|failed", "state_consistent": boolean, "observed_event_order": [string]}.'
    return "OUTPUT_SHAPE: Return one JSON object."


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


def _split_full_rows(
    tasks: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    *,
    split_names: Sequence[str] = FINAL_SPLITS,
) -> Mapping[str, Mapping[str, tuple[Mapping[str, Any], ...]]]:
    split_label_rows = _split_labels_by_status_source(labels, split_names=split_names)
    feature_by_label = _feature_by_label_id(features)
    task_by_id = _task_by_id(_unique_tasks_by_id(tasks))
    result: dict[str, dict[str, tuple[Mapping[str, Any], ...]]] = {}
    for split in split_names:
        ordered_labels = tuple(split_label_rows[split])
        ordered_tasks = _tasks_for_labels(ordered_labels, task_by_id)
        ordered_features = tuple(
            feature_by_label[str(label.get("label_id") or "")]
            for label in ordered_labels
            if str(label.get("label_id") or "") in feature_by_label
        )
        result[split] = {"tasks": ordered_tasks, "labels": ordered_labels, "features": ordered_features}
    return result


def _split_labels_by_status_source(
    labels: Sequence[Mapping[str, Any]],
    *,
    split_names: Sequence[str] = FINAL_SPLITS,
) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    unique = _sort_labels_for_output(_unique_labels_by_id(labels))
    split_rows: dict[str, list[Mapping[str, Any]]] = {split: [] for split in split_names}
    for status in (True, False, None):
        status_labels = tuple(
            label
            for label in unique
            if (
                label.get("is_utility_preserved") is status
                if status is not None
                else label.get("is_utility_preserved") not in {True, False}
            )
        )
        status_split_rows = _split_one_label_status(status_labels, split_names=split_names)
        for split in split_names:
            split_rows[split].extend(status_split_rows[split])
    return {split: _sort_labels_for_output(rows) for split, rows in split_rows.items()}


def _split_one_label_status(
    labels: Sequence[Mapping[str, Any]],
    *,
    split_names: Sequence[str],
) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    split_rows: dict[str, list[Mapping[str, Any]]] = {split: [] for split in split_names}
    if not labels:
        return {split: () for split in split_names}
    for source in DATASET_SOURCES:
        source_labels = tuple(label for label in labels if str(label.get("dataset_source") or "") == source)
        for index, label in enumerate(source_labels):
            split = _split_for_source_index(index, len(source_labels), split_names=split_names)
            split_rows[split].append(label)
    desired_counts = _desired_split_counts(len(labels), split_names=split_names)
    _rebalance_split_rows(split_rows, desired_counts)
    return {split: tuple(rows) for split, rows in split_rows.items()}


def _desired_split_counts(total: int, *, split_names: Sequence[str]) -> Mapping[str, int]:
    train_split, valid_split, test_split = tuple(split_names)
    train_count = int(total * 0.8)
    valid_count = int(total * 0.1)
    test_count = total - train_count - valid_count
    if total >= 10:
        return {train_split: train_count, valid_split: valid_count, test_split: test_count}
    if total == 1:
        return {train_split: 1, valid_split: 0, test_split: 0}
    if total == 2:
        return {train_split: 1, valid_split: 0, test_split: 1}
    return {train_split: max(1, total - 2), valid_split: 1, test_split: 1}


def _rebalance_split_rows(split_rows: dict[str, list[Mapping[str, Any]]], desired_counts: Mapping[str, int]) -> None:
    while True:
        overfull = [
            split
            for split, rows in split_rows.items()
            if len(rows) > int(desired_counts.get(split, 0))
        ]
        underfull = [
            split
            for split, rows in split_rows.items()
            if len(rows) < int(desired_counts.get(split, 0))
        ]
        if not overfull or not underfull:
            break
        source = max(overfull, key=lambda split: len(split_rows[split]) - int(desired_counts.get(split, 0)))
        target = max(underfull, key=lambda split: int(desired_counts.get(split, 0)) - len(split_rows[split]))
        split_rows[target].append(split_rows[source].pop())


def _tasks_for_labels(labels: Sequence[Mapping[str, Any]], task_by_id: Mapping[str, Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    tasks: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for label in labels:
        task_id = str(label.get("task_id") or "")
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        tasks.append(task_by_id.get(task_id) or _task_stub_from_label(label))
    return tuple(tasks)


def _split_for_source_index(index: int, count: int, *, split_names: Sequence[str] = FINAL_SPLITS) -> str:
    train_split, valid_split, test_split = tuple(split_names)
    if count >= 10:
        train_cut = max(1, int(count * 0.8))
        valid_cut = max(train_cut + 1, int(count * 0.9))
        if index < train_cut:
            return train_split
        if index < valid_cut:
            return valid_split
        return test_split
    if index == count - 1:
        return test_split
    if index == count - 2:
        return valid_split
    return train_split


def _write_split_outputs(paths: Mapping[str, Any], split_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> None:
    for split in split_rows.keys():
        rows = split_rows[split]
        _write_jsonl(paths["tasks"][split], rows["tasks"])
        _write_jsonl(paths["labels"][split], rows["labels"])
        _write_jsonl(paths["features"][split], rows["features"])
        _write_jsonl(paths["feature_aliases"][split], rows["features"])


def _load_existing_full_rows(output: Path) -> Mapping[str, tuple[dict[str, Any], ...]]:
    tasks: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    for split in FINAL_SPLITS:
        tasks.extend(_read_jsonl_safe(output / "tasks" / split / "utility_tasks.jsonl"))
        labels.extend(_read_jsonl_safe(output / "labels" / split / "utility_labels.jsonl"))
        features.extend(_read_jsonl_safe(output / "features" / split / "training_features.jsonl"))
    return {"tasks": tuple(tasks), "labels": tuple(labels), "features": tuple(features)}


def _label_dedupe_key(label: Mapping[str, Any]) -> tuple[str, str]:
    return str(label.get("task_id") or ""), str(label.get("candidate_strategy") or "")


def _preserved_counts(labels: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    return {
        "true": sum(1 for label in labels if label.get("is_utility_preserved") is True),
        "false": sum(1 for label in labels if label.get("is_utility_preserved") is False),
        "skipped": sum(1 for label in labels if label.get("is_utility_preserved") not in {True, False}),
    }


def _mark_expanded_task(task: Mapping[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(dict(task))
    cloned["_local_dsapi_prompt_sensitive"] = True
    source_metadata = cloned.get("source_metadata") if isinstance(cloned.get("source_metadata"), Mapping) else {}
    cloned["source_metadata"] = {**dict(source_metadata), "expanded_real_prompt_sensitive": True}
    return cloned


def _expanded_report_file_names() -> Mapping[str, str]:
    return {
        "full_build_summary": "expanded_build_summary.json",
        "cost_summary": "expanded_cost_summary.json",
        "quality_report": "expanded_quality_report.json",
        "dataset_card": "expanded_dataset_card.md",
        "training_usage": "expanded_training_usage.md",
    }


def _full_cost_summary(cost_summary: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "schema_version": "utility-validator-full-cost-summary-v1",
        "api_call_count": _int(cost_summary.get("api_call_count")),
        "max_api_calls": _int(cost_summary.get("max_api_calls")),
        "model": cost_summary.get("model"),
        "model_is_v4_flash": bool(cost_summary.get("model_is_v4_flash")),
        "model_is_pro": bool(cost_summary.get("model_is_pro")),
        "api_key_stored": bool(cost_summary.get("api_key_stored")),
        "total_input_tokens": _int(cost_summary.get("total_input_tokens")),
        "total_output_tokens": _int(cost_summary.get("total_output_tokens")),
        "total_cached_tokens": _int(cost_summary.get("total_cached_tokens")),
        "cached_tokens": _int(cost_summary.get("cached_tokens")),
        "total_tokens": _int(cost_summary.get("total_tokens")),
        "estimated_cost_usd": float(cost_summary.get("estimated_cost_usd") or 0.0),
        "provider_cost_usd": float(cost_summary.get("provider_cost_usd") or 0.0),
        "price_input_per_million": float(cost_summary.get("price_input_per_million") or 0.0),
        "price_output_per_million": float(cost_summary.get("price_output_per_million") or 0.0),
        "latency_seconds_total": float(cost_summary.get("latency_seconds_total") or 0.0),
        "latency_seconds_avg": float(cost_summary.get("latency_seconds_avg") or 0.0),
        "calls_recorded": len(tuple(cost_summary.get("calls") or ())),
    }


def _full_quality_report(
    *,
    split_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    checkpoint_reports: Sequence[Mapping[str, Any]],
    anomalous_batches: Sequence[Mapping[str, Any]],
    output_root: Path,
    created_at: str,
    split_names: Sequence[str] = FINAL_SPLITS,
    scan_report_names: Mapping[str, str] | None = None,
    schema_version: str = "utility-validator-full-quality-report-v1",
) -> Mapping[str, Any]:
    labels = tuple(label for split in split_names for label in split_rows[split]["labels"])
    features = tuple(feature for split in split_names for feature in split_rows[split]["features"])
    source_counts = Counter(str(label.get("dataset_source")) for label in labels)
    preserved_counts = Counter(str(label.get("is_utility_preserved")) for label in labels)
    failure_counts = Counter(str(label.get("failure_type") or "none") for label in labels)
    strategy_counts = Counter(str(label.get("candidate_strategy") or "unknown") for label in labels)
    strategy_false_counts = Counter(
        str(label.get("candidate_strategy") or "unknown")
        for label in labels
        if label.get("is_utility_preserved") is False
    )
    trainable_count = sum(1 for label in labels if label.get("is_utility_preserved") in {True, False})
    scan_report = _scan_prompt_safe_outputs(
        output_root,
        split_names=split_names,
        report_names=scan_report_names,
    )
    return {
        "schema_version": schema_version,
        "created_at": created_at,
        "label_count": len(labels),
        "feature_count": len(features),
        "trainable_label_count": trainable_count,
        "feature_count_matches_trainable_labels": len(features) == trainable_count,
        "source_counts": dict(sorted(source_counts.items())),
        "is_utility_preserved_counts": dict(sorted(preserved_counts.items())),
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "candidate_strategy_counts": dict(sorted(strategy_counts.items())),
        "candidate_strategy_false_counts": dict(sorted(strategy_false_counts.items())),
        "label_source_counts": dict(sorted(Counter(str(label.get("label_source")) for label in labels).items())),
        "prompt_text_saved": _has_prompt_text(labels) or _has_prompt_text(features),
        "scan_report": scan_report,
        "checkpoint_count": len(checkpoint_reports),
        "checkpoint_passed_count": sum(1 for report in checkpoint_reports if report.get("passed") is True),
        "anomalous_batch_count": len(anomalous_batches),
        "anomalous_batches": tuple(anomalous_batches),
        "original_failed_skipped_not_in_features": _original_failed_skipped_not_in_features(labels, features),
        "all_sources_present": all(source in source_counts for source in DATASET_SOURCES),
        "ready_for_baseline_training": (
            bool(labels)
            and len(features) == trainable_count
            and not anomalous_batches
            and all(label.get("label_source") == "dsapi_execution_oracle" for label in labels)
            and not scan_report["has_hits"]
        ),
    }


def _full_build_summary(
    *,
    source: str,
    max_tasks: int,
    batch_size: int,
    checkpoint_api_calls: int,
    max_api_calls: int,
    include_text: bool,
    created_at: str,
    elapsed_seconds: float,
    split_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    cost_summary: Mapping[str, Any],
    quality_report: Mapping[str, Any],
    checkpoint_reports: Sequence[Mapping[str, Any]],
    anomalous_batches: Sequence[Mapping[str, Any]],
    output_paths: Mapping[str, Any],
    source_reports: Mapping[str, Any],
    total_cost_at_last_checkpoint: float,
    split_names: Sequence[str] = FINAL_SPLITS,
    mode: str = "train_real",
    schema_version: str = "utility-validator-full-build-summary-v1",
    extra_fields: Mapping[str, Any] | None = None,
    exception_policy: str = "anomalous batches are written under reports/full_checkpoints and not merged",
) -> Mapping[str, Any]:
    split_counts = {
        split: {
            "task_count": len(split_rows[split]["tasks"]),
            "label_count": len(split_rows[split]["labels"]),
            "feature_count": len(split_rows[split]["features"]),
        }
        for split in split_names
    }
    labels = tuple(label for split in split_names for label in split_rows[split]["labels"])
    features = tuple(feature for split in split_names for feature in split_rows[split]["features"])
    summary = {
        "schema_version": schema_version,
        "created_at": created_at,
        "source": source,
        "mode": mode,
        "backend": "dsapi",
        "api_model": cost_summary.get("model"),
        "api_model_is_v4_flash": cost_summary.get("model_is_v4_flash"),
        "api_model_is_pro": cost_summary.get("model_is_pro"),
        "max_tasks": max_tasks,
        "batch_size": batch_size,
        "checkpoint_api_calls": checkpoint_api_calls,
        "max_api_calls": max_api_calls,
        "split_counts": split_counts,
        "label_count": len(labels),
        "feature_count": len(features),
        "source_counts": quality_report.get("source_counts"),
        "is_utility_preserved_counts": quality_report.get("is_utility_preserved_counts"),
        "failure_type_counts": quality_report.get("failure_type_counts"),
        "cost_summary": cost_summary,
        "quality_report_path": str(output_paths["reports"]["quality_report"]),
        "cost_summary_path": str(output_paths["reports"]["cost_summary"]),
        "dataset_card_path": str(output_paths["reports"]["dataset_card"]),
        "training_usage_path": str(output_paths["reports"]["training_usage"]),
        "outputs": _stringify_output_paths(output_paths),
        "prompt_text_saved": quality_report.get("prompt_text_saved"),
        "scan_report": quality_report.get("scan_report"),
        "checkpoint_count": len(checkpoint_reports),
        "checkpoint_reports": tuple(checkpoint_reports),
        "anomalous_batch_count": len(anomalous_batches),
        "anomalous_batches": tuple(anomalous_batches),
        "exception_policy": exception_policy,
        "source_reports": source_reports,
        "include_text": include_text,
        "ready_for_baseline_training": quality_report.get("ready_for_baseline_training"),
        "elapsed_seconds": elapsed_seconds,
        "total_cost_at_last_checkpoint": total_cost_at_last_checkpoint,
        "pytest_report": None,
    }
    if extra_fields:
        summary.update(dict(extra_fields))
    return summary


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


def update_full_build_summary_with_pytest(
    *,
    output_root: str | Path,
    pytest_q_passed: bool,
    pytest_report: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    output = Path(output_root)
    summary_path = output / "reports" / "full_build_summary.json"
    summary = dict(_read_json_safe(summary_path))
    summary["pytest_report"] = {
        "pytest_q_passed": pytest_q_passed,
        **dict(pytest_report or {}),
    }
    summary["ready_for_baseline_training"] = bool(summary.get("ready_for_baseline_training")) and pytest_q_passed
    _write_json(summary_path, summary)
    return summary


def update_expanded_build_summary_with_pytest(
    *,
    output_root: str | Path,
    pytest_q_passed: bool,
    pytest_report: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    output = Path(output_root)
    summary_path = output / "reports" / "expanded_build_summary.json"
    summary = dict(_read_json_safe(summary_path))
    summary["pytest_report"] = {
        "pytest_q_passed": pytest_q_passed,
        **dict(pytest_report or {}),
    }
    summary["ready_for_baseline_training"] = bool(summary.get("ready_for_baseline_training")) and pytest_q_passed
    _write_json(summary_path, summary)
    return summary


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


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


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


def _scan_prompt_safe_outputs(
    output_root: Path,
    *,
    split_names: Sequence[str] = FINAL_SPLITS,
    report_names: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    scan_paths = tuple(output_root / "labels" / split for split in split_names) + tuple(
        output_root / "features" / split for split in split_names
    ) + (
        output_root / "reports" / ("expanded_checkpoints" if any(str(split).startswith("expanded_") for split in split_names) else "full_checkpoints"),
    )
    report_file_names = {
        "full_build_summary": "full_build_summary.json",
        "cost_summary": "cost_summary.json",
        "quality_report": "quality_report.json",
        "dataset_card": "dataset_card.md",
        "training_usage": "training_usage.md",
    }
    if report_names is not None:
        report_file_names.update(dict(report_names))
    report_files = (
        output_root / "reports" / report_file_names["full_build_summary"],
        output_root / "reports" / report_file_names["cost_summary"],
        output_root / "reports" / report_file_names["quality_report"],
        output_root / "reports" / report_file_names["dataset_card"],
        output_root / "reports" / report_file_names["training_usage"],
    )
    literal_patterns = (
        "def has_close_elements",
        "SMOKE-PRIVATE-TOKEN-7",
        "ALPHA-PRIVATE",
        "Internal account token",
        "Auditor note",
        "PROMPT_UNDER_TEST",
        "ORACLE_OUTPUT_CONTRACT",
    )
    secret_patterns = (re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{15,}\b"),)
    hits: list[dict[str, Any]] = []
    for root in scan_paths:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".md", ".txt"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pattern in literal_patterns:
                if pattern in text:
                    hits.append({"path": str(path), "pattern": pattern})
            for pattern in secret_patterns:
                if pattern.search(text):
                    hits.append({"path": str(path), "pattern": pattern.pattern})
    for path in report_files:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in literal_patterns:
            if pattern in text:
                hits.append({"path": str(path), "pattern": pattern})
        for pattern in secret_patterns:
            if pattern.search(text):
                hits.append({"path": str(path), "pattern": pattern.pattern})
    return {
        "has_hits": bool(hits),
        "hit_count": len(hits),
        "hits": tuple(hits[:20]),
        "scanned_paths": tuple(str(path) for path in scan_paths + report_files),
    }


def _jsonl_readability_report(output_root: Path) -> Mapping[str, Any]:
    paths = tuple(
        path
        for base in (
            output_root / "labels",
            output_root / "features",
            output_root / "tasks",
        )
        if base.exists()
        for path in base.rglob("*.jsonl")
    )
    errors: list[dict[str, Any]] = []
    for path in paths:
        try:
            _read_jsonl_safe(path)
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(path), "error": str(exc)})
    return {"readable": not errors, "checked_file_count": len(paths), "errors": tuple(errors[:20])}


def _stringify_output_paths(paths: Mapping[str, Any]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in paths.items():
        if isinstance(value, Mapping):
            result[key] = {inner_key: str(inner_value) for inner_key, inner_value in value.items()}
        else:
            result[key] = str(value)
    return result


def _dataset_card_markdown(summary: Mapping[str, Any]) -> str:
    split_counts = summary.get("split_counts") if isinstance(summary.get("split_counts"), Mapping) else {}
    return "\n".join(
        [
            "# Utility Validator Full Real Dataset",
            "",
            f"Created: {summary.get('created_at')}",
            f"Model: {summary.get('api_model')}",
            "Label source: dsapi_execution_oracle",
            "",
            "## Splits",
            "",
            json.dumps(split_counts, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "## Sources",
            "",
            json.dumps(summary.get("source_counts") or {}, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "## Labels",
            "",
            "The target label is `is_utility_preserved`. Use only rows where this field is true or false for supervised training.",
            "",
            "## Safety",
            "",
            f"Prompt text saved: {summary.get('prompt_text_saved')}",
            "API keys are not stored in dataset artifacts.",
        ]
    )


def _training_usage_markdown(summary: Mapping[str, Any]) -> str:
    outputs = summary.get("outputs") if isinstance(summary.get("outputs"), Mapping) else {}
    return "\n".join(
        [
            "# Training Usage",
            "",
            "## Dataset Location",
            "",
            f"Labels: `{outputs.get('labels')}`",
            f"Features: `{outputs.get('features')}`",
            "",
            "## Files",
            "",
            "- `features/train/train_features.jsonl`, `features/valid/valid_features.jsonl`, `features/test/test_features.jsonl`: direct training feature aliases.",
            "- `features/*/training_features.jsonl`: same feature rows under the existing dataset layout.",
            "- `labels/*/utility_labels.jsonl`: oracle labels and run metadata.",
            "",
            "## Feature Fields",
            "",
            "Use structured fields such as `dataset_source`, `scenario_type`, `candidate_strategy`, `source_scopes`, `target_scopes`, `risk_tags`, `dependency_notes`, `moved_block_count`, `movement_distance_summary`, `estimated_cache_gain`, `cached_tokens_delta`, and `hard_warning_count` as model inputs.",
            "",
            "## Training Label",
            "",
            "`is_utility_preserved` is the supervised target. `true` means reordered prompt preserved utility; `false` means original passed but reordered failed.",
            "",
            "## Filtering",
            "",
            "Filter out rows where `is_utility_preserved` is null. These are skipped samples, usually because the original prompt failed and should not enter supervised training.",
            "",
            "## Fake Data Guard",
            "",
            "Use only labels with `label_source == dsapi_execution_oracle`; never mix `fake_smoke_oracle` rows into train/valid/test.",
            "",
            "## Reading Splits",
            "",
            "Read JSONL line by line, parse each line as JSON, then join feature rows to labels by `label_id` if label metadata is needed.",
            "",
            "## Baseline Training",
            "",
            "For a first Utility Validator baseline, train a binary classifier on non-skipped feature rows with `is_utility_preserved` as the target. Keep valid/test untouched for model selection and final evaluation.",
            "",
            "## Limitations",
            "",
            "This dataset is produced from a compact source set with deterministic prompt variants. It validates the real DS API labeling chain, but broader source diversity may be needed before strong generalization claims.",
        ]
    )


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


def _schema_parse_error_count(labels: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    parse_markers = ("output_is_not_valid_json", "parse", "invalid_json", "json_decode")
    for label in labels:
        if label.get("failure_type") != "json_schema_fail":
            continue
        reordered = (label.get("oracle_reports") or {}).get("reordered") if isinstance(label.get("oracle_reports"), Mapping) else {}
        reason = str((reordered or {}).get("reason") or label.get("utility_status") or "").lower()
        if any(marker in reason for marker in parse_markers):
            count += 1
    return count


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
