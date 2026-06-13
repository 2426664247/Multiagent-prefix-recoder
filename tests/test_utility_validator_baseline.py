from __future__ import annotations

import json
from pathlib import Path

import pytest
from autogen_core.models import LLMMessage, SystemMessage

from autogen_prefix_tree import CacheUtilityValidator, HierarchicalPrefixPlanner, LocalPromptCompiler, rewrite_messages
from autogen_prefix_tree.planner_ranker_baseline import (
    TARGET_FIELD as PLANNER_TARGET_FIELD,
    PlannerRankerModel,
    load_planner_ranker_dataset,
    train_planner_ranker_baseline_v0,
)

from autogen_prefix_tree.utility_validator_baseline import (
    LEAKAGE_FIELDS,
    PROMPT_TEXT_FIELDS,
    UtilityValidatorModel,
    discover_expanded_dataset_paths,
    load_expanded_utility_validator_dataset,
    train_utility_validator_baseline_v0,
)


def test_real_utility_validator_baseline_v0_loads_predicts_evaluates_and_reads_metrics() -> None:
    model_path = Path("artifacts/utility_validator/baseline_v0/model.pkl")
    metrics_path = Path("artifacts/utility_validator/baseline_v0/metrics.json")
    schema_path = Path("artifacts/utility_validator/baseline_v0/feature_schema.json")
    assert model_path.exists()
    assert metrics_path.exists()
    assert schema_path.exists()

    model = UtilityValidatorModel.load(model_path)
    dataset = load_expanded_utility_validator_dataset(
        repo_root=".",
        dataset_root="datasets/utility_validator",
    )
    row = dataset.splits["expanded_test"].rows[0]
    prediction = model.predict(row)
    assert isinstance(prediction["is_utility_preserved"], bool)
    assert 0.0 <= prediction["confidence"] <= 1.0

    compile_result, plan = _runtime_plan()
    cache_utility_estimate = CacheUtilityValidator()._estimate_cache_utility(compile_result, plan, None)
    evaluation = model.evaluate(
        original_messages=compile_result.messages,
        rewritten_messages=rewrite_messages(compile_result, plan),
        compile_result=compile_result,
        plan=plan,
        cache_utility_estimate=cache_utility_estimate,
    )
    assert evaluation.prompt_safe is True
    assert evaluation.model_name == model.model_name
    assert evaluation.is_utility_preserved in {True, False}

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    raw_fields = set(schema["raw_input_fields"]["categorical"])
    raw_fields.update(schema["raw_input_fields"]["multi_value"])
    raw_fields.update(schema["raw_input_fields"]["numeric"])
    encoded_feature_names = "\n".join(schema["encoded_feature_names"])

    required_data_fields = {
        field
        for field in raw_fields
        if field.split(".", 1)[0] not in {"placement_changes", "cache_gain_report"}
    }
    assert all(_field_exists(row, field) for field in required_data_fields)
    assert {"placement_changes.count", "cache_gain_report.estimated_cache_gain"}.issubset(raw_fields)
    assert schema["prompt_text_used"] is False
    assert metrics["prompt_text_used"] is False
    assert metrics["ds_api_called"] is False
    assert set(LEAKAGE_FIELDS).isdisjoint(raw_fields)
    assert all(field not in encoded_feature_names for field in LEAKAGE_FIELDS)
    assert all(field not in encoded_feature_names for field in PROMPT_TEXT_FIELDS)


def test_loads_real_expanded_train_valid_test_and_excludes_skipped() -> None:
    dataset = load_expanded_utility_validator_dataset(
        repo_root=".",
        dataset_root="datasets/utility_validator",
    )

    assert set(dataset.splits) == {"expanded_train", "expanded_valid", "expanded_test"}
    assert len(dataset.splits["expanded_train"].rows) == 160
    assert len(dataset.splits["expanded_valid"].rows) == 20
    assert len(dataset.splits["expanded_test"].rows) == 20
    assert dataset.skipped_label_count == 101
    assert dataset.trainable_count == 200
    assert dataset.splits["expanded_train"].feature_distribution == {"False": 40, "True": 120}
    assert dataset.splits["expanded_valid"].feature_distribution == {"False": 5, "True": 15}
    assert dataset.splits["expanded_test"].feature_distribution == {"False": 5, "True": 15}
    assert "None" in dataset.splits["expanded_train"].label_distribution
    assert None not in dataset.splits["expanded_train"].y


def test_refuses_smoke_fake_paths_and_fake_labels(tmp_path: Path) -> None:
    smoke_root = tmp_path / "smoke" / "utility_validator"
    _write_minimal_expanded_dataset(smoke_root)

    with pytest.raises(ValueError, match="smoke/fake"):
        discover_expanded_dataset_paths(repo_root=tmp_path, dataset_root=smoke_root)

    dataset_root = tmp_path / "datasets" / "utility_validator"
    _write_minimal_expanded_dataset(dataset_root, label_source="fake_smoke_oracle")

    with pytest.raises(ValueError, match="fake_smoke_oracle"):
        load_expanded_utility_validator_dataset(repo_root=tmp_path, dataset_root=dataset_root)


def test_leakage_fields_and_prompt_text_are_not_feature_inputs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets" / "utility_validator"
    _write_minimal_expanded_dataset(dataset_root)

    result = train_utility_validator_baseline_v0(
        repo_root=tmp_path,
        dataset_root=dataset_root,
        output_dir=tmp_path / "artifacts" / "utility_validator" / "baseline_v0",
        logistic_iterations=40,
        random_forest_estimators=4,
    )

    schema = result["feature_schema"]
    encoded_feature_names = "\n".join(schema["encoded_feature_names"])
    raw_fields = set(schema["raw_input_fields"]["categorical"])
    raw_fields.update(schema["raw_input_fields"]["multi_value"])
    raw_fields.update(schema["raw_input_fields"]["numeric"])

    assert schema["prompt_text_used"] is False
    assert set(LEAKAGE_FIELDS).isdisjoint(raw_fields)
    assert all(field not in encoded_feature_names for field in LEAKAGE_FIELDS)
    assert all(field not in encoded_feature_names for field in PROMPT_TEXT_FIELDS)
    assert "is_utility_preserved" not in encoded_feature_names


def test_model_trains_saves_loads_predicts_and_writes_metrics(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets" / "utility_validator"
    output_dir = tmp_path / "artifacts" / "utility_validator" / "baseline_v0"
    _write_minimal_expanded_dataset(dataset_root)

    result = train_utility_validator_baseline_v0(
        repo_root=tmp_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
        logistic_iterations=60,
        random_forest_estimators=5,
    )

    model_path = Path(result["model_path"])
    metrics_path = output_dir / "metrics.json"
    assert model_path.exists()
    assert (output_dir / "feature_schema.json").exists()
    assert metrics_path.exists()
    assert (output_dir / "training_report.md").exists()

    model = UtilityValidatorModel.load(model_path)
    row = load_expanded_utility_validator_dataset(repo_root=tmp_path, dataset_root=dataset_root).splits["expanded_test"].rows[0]
    prediction = model.predict(row)
    assert set(prediction) >= {"is_utility_preserved", "confidence", "failure_type", "reason"}
    assert isinstance(prediction["is_utility_preserved"], bool)
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert prediction["failure_type"] is None
    assert prediction["reason"] == "baseline model prediction from structured placement features"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["ds_api_called"] is False
    assert metrics["prompt_text_used"] is False
    assert set(metrics["models"]) == {"logistic_regression_balanced", "random_forest_balanced"}
    assert "confusion_matrix" in metrics["selected_valid_metrics"]
    assert "negative_recall" in metrics["selected_test_metrics"]
    assert "false_negative_rate" in metrics["selected_test_metrics"]
    assert "per_source" in metrics["selected_test_metrics"]


def test_training_does_not_call_ds_api_or_import_dataset_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_root = tmp_path / "datasets" / "utility_validator"
    _write_minimal_expanded_dataset(dataset_root)

    def fail_urlopen(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("DS API/network should not be called")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    train_utility_validator_baseline_v0(
        repo_root=tmp_path,
        dataset_root=dataset_root,
        output_dir=tmp_path / "artifacts" / "utility_validator" / "baseline_v0",
        logistic_iterations=20,
        random_forest_estimators=3,
    )


def test_planner_ranker_trains_saves_loads_predicts_and_excludes_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "datasets" / "utility_validator"
    output_dir = tmp_path / "artifacts" / "planner_ranker" / "baseline_v0"
    _write_minimal_expanded_dataset(dataset_root)

    def fail_urlopen(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("DS API/network should not be called")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    result = train_planner_ranker_baseline_v0(
        repo_root=tmp_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
        logistic_iterations=40,
        random_forest_estimators=4,
    )

    assert Path(result["model_path"]).exists()
    assert (output_dir / "feature_schema.json").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "training_report.md").exists()

    model = PlannerRankerModel.load(result["model_path"])
    dataset = load_planner_ranker_dataset(repo_root=tmp_path, dataset_root=dataset_root)
    row = dataset.splits["expanded_test"].rows[0]
    prediction = model.predict(row)
    assert set(prediction) >= {
        "is_priority_candidate",
        "priority_probability",
        "confidence",
        "priority_score",
        "threshold",
        "model_name",
    }
    assert isinstance(prediction["is_priority_candidate"], bool)
    assert 0.0 <= prediction["priority_probability"] <= 1.0
    ranking = model.rank(dataset.splits["expanded_test"].rows)
    assert len(ranking) == len(dataset.splits["expanded_test"].rows)
    assert ranking[0]["priority_score"] >= ranking[-1]["priority_score"]

    schema = result["feature_schema"]
    raw_fields = set(schema["raw_input_fields"]["categorical"])
    raw_fields.update(schema["raw_input_fields"]["multi_value"])
    raw_fields.update(schema["raw_input_fields"]["numeric"])
    encoded_feature_names = "\n".join(schema["encoded_feature_names"])
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert schema["target_field"] == PLANNER_TARGET_FIELD
    assert schema["prompt_text_used"] is False
    assert schema["ds_api_called"] is False
    assert metrics["prompt_text_used"] is False
    assert metrics["ds_api_called"] is False
    assert set(LEAKAGE_FIELDS).isdisjoint(raw_fields)
    assert all(field not in encoded_feature_names for field in LEAKAGE_FIELDS)
    assert all(field not in encoded_feature_names for field in PROMPT_TEXT_FIELDS)
    assert "is_utility_preserved" not in encoded_feature_names
    assert PLANNER_TARGET_FIELD not in encoded_feature_names


def test_planner_ranker_rejects_prompt_text_body_rows(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets" / "utility_validator"
    _write_minimal_expanded_dataset(dataset_root, include_prompt_text=True)

    with pytest.raises(ValueError, match="prompt text/body"):
        load_planner_ranker_dataset(repo_root=tmp_path, dataset_root=dataset_root)


def test_prompt_text_body_rows_are_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets" / "utility_validator"
    _write_minimal_expanded_dataset(dataset_root, include_prompt_text=True)

    with pytest.raises(ValueError, match="prompt text/body"):
        load_expanded_utility_validator_dataset(repo_root=tmp_path, dataset_root=dataset_root)


def _write_minimal_expanded_dataset(
    dataset_root: Path,
    *,
    label_source: str = "dsapi_execution_oracle",
    include_prompt_text: bool = False,
) -> None:
    for split, feature_name in (
        ("expanded_train", "train_features.jsonl"),
        ("expanded_valid", "valid_features.jsonl"),
        ("expanded_test", "test_features.jsonl"),
    ):
        feature_dir = dataset_root / "features" / split
        label_dir = dataset_root / "labels" / split
        feature_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        feature_rows = []
        label_rows = []
        labeled_count = 8 if split == "expanded_train" else 4
        for index in range(labeled_count):
            target = index % 2 == 0
            label_id = f"{split}_label_{index}"
            feature_rows.append(_feature_row(label_id=label_id, index=index, target=target))
            label_rows.append(_label_row(label_id=label_id, target=target, label_source=label_source))

        label_rows.append(_label_row(label_id=f"{split}_skipped", target=None, label_source=label_source))
        if include_prompt_text:
            feature_rows[0]["original_prompt_text"] = "raw prompt body must not be used"

        _write_jsonl(feature_dir / feature_name, feature_rows)
        _write_jsonl(feature_dir / "training_features.jsonl", feature_rows)
        _write_jsonl(label_dir / "utility_labels.jsonl", label_rows)


def _feature_row(*, label_id: str, index: int, target: bool) -> dict[str, object]:
    strategy = "conservative" if index % 2 == 0 else "aggressive"
    source = "source_a" if index % 3 else "source_b"
    return {
        "label_id": label_id,
        "sample_id": f"sample_{label_id}",
        "dataset_source": source,
        "scenario_type": "unit_test",
        "candidate_strategy": strategy,
        "source_scopes": ["global", "agent_local"],
        "target_scopes": ["global"] if target else ["agent_local"],
        "risk_tags": ["shared_context"] if target else ["private_memory", "latest_user_instruction"],
        "dependency_notes": [] if target else ["private_memory_boundary"],
        "moved_block_count": 1 if target else 4,
        "movement_distance_summary": {
            "max_abs_distance": 1 if target else 5,
            "moved_position_count": 1 if target else 4,
            "sum_abs_distance": 1 if target else 9,
        },
        "global_prefix_tokens_or_chars": 80 if target else 30,
        "subgroup_prefix_tokens_or_chars": 0,
        "estimated_cache_gain": 120.0 if target else 60.0,
        "hard_warning_count": 0 if target else 2,
        "placement_changes": [
            {
                "moved": not target,
                "target_scope": "global" if target else "agent_local",
                "cache_contribution": 20.0,
                "placement_score": 0.8 if target else 0.2,
                "dependency_notes": [] if target else ["private_memory_boundary"],
                "risk_tags": ["shared_context"] if target else ["private_memory"],
            }
        ],
        "cache_gain_report": {
            "longest_common_prefix_tokens": 80 if target else 30,
            "global_prefix_tokens": 80 if target else 30,
            "subgroup_prefix_tokens": 0,
            "estimated_cache_gain": 120.0 if target else 60.0,
            "node_cache_contributions": {
                "root": {
                    "token_len": 80 if target else 30,
                    "cache_contribution": 120.0 if target else 60.0,
                }
            },
            "placement_cache_contributions": {"p0": 20.0},
        },
        "prompt_text_included": False,
        "is_utility_preserved": target,
        "failure_type": "none" if target else "unit_test_fail",
        "original_run_status": "passed",
        "reordered_run_status": "passed" if target else "failed",
        "label_source": "should_not_be_used",
        "api_model": "deepseek-v4-flash",
        "input_tokens": 999,
        "output_tokens": 111,
        "cached_tokens": 222,
        "latency": 1.23,
        "estimated_cost": 0.001,
        "whether_label_is_utility_verified": True,
    }


def _label_row(*, label_id: str, target: bool | None, label_source: str) -> dict[str, object]:
    return {
        "label_id": label_id,
        "label_source": label_source,
        "is_utility_preserved": target,
        "prompt_text_included": False,
        "api_model": "deepseek-v4-flash",
        "failure_type": "none" if target is not False else "unit_test_fail",
        "original_run_status": "passed" if target is not None else "failed",
        "reordered_run_status": "passed" if target else "failed",
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _runtime_plan():
    compiler = LocalPromptCompiler()
    planner = HierarchicalPrefixPlanner()
    cold = compiler.compile(_runtime_messages("planner"), session_id="real-utility-validator-artifact")
    planner.plan(cold, session_id="real-utility-validator-artifact")
    warm = compiler.compile(_runtime_messages("engineer"), session_id="real-utility-validator-artifact")
    plan = planner.plan(warm, session_id="real-utility-validator-artifact")
    return warm, plan


def _runtime_messages(agent: str) -> tuple[LLMMessage, ...]:
    return (
        SystemMessage(
            content="\n\n".join(
                [
                    "ROLE_SPECIFIC_INSTRUCTION_START\n"
                    f"AGENT_NAME: {agent}\n"
                    f"You are {agent}.\n"
                    "ROLE_SPECIFIC_INSTRUCTION_END",
                    "USER_TASK_START\nVerify the utility validator artifact.\nUSER_TASK_END",
                    "SHARED_GROUPCHAT_CONTEXT_START\nShared artifact verification context.\nSHARED_GROUPCHAT_CONTEXT_END",
                    "TOOL_SCHEMA_START\nshared_tool(x: string) -> string\nTOOL_SCHEMA_END",
                    "OUTPUT_FORMAT_START\nReturn a concise status.\nOUTPUT_FORMAT_END",
                ]
            )
        ),
    )


def _field_exists(row: dict[str, object], dotted_field: str) -> bool:
    if "." in dotted_field and dotted_field.split(".", 1)[0] in {"placement_changes", "cache_gain_report"}:
        return dotted_field.split(".", 1)[0] in row
    current: object = row
    for part in dotted_field.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True
