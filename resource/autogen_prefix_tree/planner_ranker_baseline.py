from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .utility_validator_baseline import (
    LEAKAGE_FIELDS,
    PROMPT_TEXT_FIELDS,
    LogisticRegressionBaseline,
    RandomForestBaseline,
    StructuredFeatureEncoder,
    evaluate_predictions,
    load_expanded_utility_validator_dataset,
)


TARGET_FIELD = "is_priority_candidate"


@dataclass(frozen=True)
class PlannerRankerSplit:
    split: str
    rows: tuple[dict[str, Any], ...]
    y: tuple[int, ...]
    utility_labels: tuple[int, ...]
    sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class PlannerRankerDataset:
    splits: Mapping[str, PlannerRankerSplit]
    cache_gain_priority_threshold: float
    trainable_count: int
    target_distribution: Mapping[str, int]
    source_distribution: Mapping[str, int]


@dataclass
class PlannerRankerModel:
    encoder: StructuredFeatureEncoder
    estimator: Any
    threshold: float
    model_name: str
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "PlannerRankerModel":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"Expected PlannerRankerModel pickle, got {type(model).__name__}")
        return model

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle)

    def predict_proba(self, row: Mapping[str, Any]) -> float:
        matrix = self.encoder.transform([dict(row)])
        return float(self.estimator.predict_proba(matrix)[0])

    def predict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        probability = self.predict_proba(row)
        should_prioritize = probability >= self.threshold
        return {
            "is_priority_candidate": bool(should_prioritize),
            "priority_probability": float(probability),
            "confidence": float(probability if should_prioritize else 1.0 - probability),
            "priority_score": float(probability),
            "threshold": float(self.threshold),
            "reason": "planner ranker baseline prediction from structured candidate and placement features",
            "model_name": self.model_name,
        }

    def rank(self, rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        scored: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            prediction = self.predict(row)
            scored.append(
                {
                    "rank_input_index": index,
                    "candidate_id": row.get("candidate_id") or row.get("sample_id") or row.get("label_id"),
                    "dataset_source": row.get("dataset_source"),
                    "candidate_strategy": row.get("candidate_strategy"),
                    "estimated_cache_gain": _cache_gain(row),
                    **prediction,
                }
            )
        return tuple(
            sorted(
                scored,
                key=lambda item: (
                    -float(item["priority_score"]),
                    -float(item.get("estimated_cache_gain") or 0.0),
                    str(item.get("candidate_id") or ""),
                ),
            )
        )


def load_planner_ranker_dataset(
    *,
    repo_root: str | Path = ".",
    dataset_root: str | Path = "datasets/utility_validator",
) -> PlannerRankerDataset:
    utility_dataset = load_expanded_utility_validator_dataset(repo_root=repo_root, dataset_root=dataset_root)
    train_split = utility_dataset.splits["expanded_train"]
    safe_train_gains = [
        _cache_gain(row)
        for row, label in zip(train_split.rows, train_split.y)
        if label == 1 and _cache_gain(row) > 0.0
    ]
    cache_gain_threshold = float(statistics.median(safe_train_gains) if safe_train_gains else 0.0)

    splits: dict[str, PlannerRankerSplit] = {}
    target_distribution: Counter[str] = Counter()
    source_distribution: Counter[str] = Counter()
    for split_name, loaded in utility_dataset.splits.items():
        rows: list[dict[str, Any]] = []
        y: list[int] = []
        utility_labels: list[int] = []
        sample_ids: list[str] = []
        for row, utility_label, sample_id in zip(loaded.rows, loaded.y, loaded.sample_ids):
            row_copy = dict(row)
            target = _priority_target(
                utility_label=utility_label,
                estimated_cache_gain=_cache_gain(row_copy),
                cache_gain_priority_threshold=cache_gain_threshold,
            )
            rows.append(row_copy)
            y.append(target)
            utility_labels.append(int(utility_label))
            sample_ids.append(str(sample_id))
            target_distribution[str(bool(target))] += 1
            source_distribution[str(row_copy.get("dataset_source"))] += 1
        splits[split_name] = PlannerRankerSplit(
            split=split_name,
            rows=tuple(rows),
            y=tuple(y),
            utility_labels=tuple(utility_labels),
            sample_ids=tuple(sample_ids),
        )
    return PlannerRankerDataset(
        splits=splits,
        cache_gain_priority_threshold=cache_gain_threshold,
        trainable_count=sum(len(split.rows) for split in splits.values()),
        target_distribution=dict(sorted(target_distribution.items())),
        source_distribution=dict(sorted(source_distribution.items())),
    )


def train_planner_ranker_baseline_v0(
    *,
    repo_root: str | Path = ".",
    dataset_root: str | Path = "datasets/utility_validator",
    output_dir: str | Path = "artifacts/planner_ranker/baseline_v0",
    random_seed: int = 43,
    logistic_iterations: int = 3000,
    random_forest_estimators: int = 80,
) -> dict[str, Any]:
    dataset = load_planner_ranker_dataset(repo_root=repo_root, dataset_root=dataset_root)
    train = dataset.splits["expanded_train"]
    valid = dataset.splits["expanded_valid"]
    test = dataset.splits["expanded_test"]

    encoder = StructuredFeatureEncoder.fit(train.rows)
    x_train = encoder.transform(train.rows)
    y_train = np.asarray(train.y, dtype=int)
    x_valid = encoder.transform(valid.rows)
    y_valid = np.asarray(valid.y, dtype=int)
    x_test = encoder.transform(test.rows)
    y_test = np.asarray(test.y, dtype=int)

    baselines: dict[str, Any] = {
        "logistic_regression_balanced": LogisticRegressionBaseline(iterations=logistic_iterations).fit(x_train, y_train),
        "random_forest_balanced": RandomForestBaseline(
            n_estimators=random_forest_estimators,
            random_state=random_seed,
        ).fit(x_train, y_train),
    }

    model_metrics: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    for name, estimator in baselines.items():
        valid_scores = estimator.predict_proba(x_valid)
        threshold = _select_threshold(y_valid, valid_scores)
        thresholds[name] = threshold
        model_metrics[name] = {
            "threshold": threshold,
            "selection_score": _selection_score(evaluate_predictions(y_valid, valid_scores, threshold)),
            "train": _planner_metrics(train, estimator.predict_proba(x_train), threshold),
            "valid": _planner_metrics(valid, valid_scores, threshold),
            "test": _planner_metrics(test, estimator.predict_proba(x_test), threshold),
        }

    selected_model_name = sorted(
        model_metrics,
        key=lambda item: (
            model_metrics[item]["selection_score"],
            model_metrics[item]["valid"]["safe_priority_recall"],
            model_metrics[item]["valid"]["unsafe_rejection_rate"],
            item,
        ),
        reverse=True,
    )[0]

    selected_model = PlannerRankerModel(
        encoder=encoder,
        estimator=baselines[selected_model_name],
        threshold=thresholds[selected_model_name],
        model_name=f"planner_ranker_baseline_v0:{selected_model_name}",
        metadata={
            "schema_version": "planner-ranker-baseline-v0",
            "selected_model": selected_model_name,
            "threshold": thresholds[selected_model_name],
            "cache_gain_priority_threshold": dataset.cache_gain_priority_threshold,
            "target_definition": (
                "positive iff is_utility_preserved is true and estimated_cache_gain is at least the "
                "expanded_train safe positive median"
            ),
            "prompt_text_used": False,
            "ds_api_called": False,
            "leakage_fields_excluded": list(LEAKAGE_FIELDS),
        },
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.pkl"
    feature_schema_path = output / "feature_schema.json"
    metrics_path = output / "metrics.json"
    training_report_path = output / "training_report.md"

    selected_model.save(model_path)
    feature_schema = encoder.to_schema()
    feature_schema.update(
        {
            "schema_version": "planner-ranker-baseline-feature-schema-v0",
            "target_field": TARGET_FIELD,
            "target_definition": selected_model.metadata["target_definition"],
            "cache_gain_priority_threshold": dataset.cache_gain_priority_threshold,
            "excluded_leakage_fields": list(LEAKAGE_FIELDS),
            "excluded_prompt_text_fields": list(PROMPT_TEXT_FIELDS),
            "prompt_text_used": False,
            "ds_api_called": False,
        }
    )
    _write_json(feature_schema_path, feature_schema)

    split_summary = _split_summary(dataset)
    metrics = {
        "schema_version": "planner-ranker-baseline-v0-metrics",
        "dataset": split_summary,
        "models": model_metrics,
        "selected_model": selected_model_name,
        "selected_threshold": thresholds[selected_model_name],
        "selected_valid_metrics": model_metrics[selected_model_name]["valid"],
        "selected_test_metrics": model_metrics[selected_model_name]["test"],
        "target_definition": selected_model.metadata["target_definition"],
        "cache_gain_priority_threshold": dataset.cache_gain_priority_threshold,
        "leakage_fields_excluded": list(LEAKAGE_FIELDS),
        "prompt_text_used": False,
        "ds_api_called": False,
    }
    _write_json(metrics_path, metrics)
    training_report_path.write_text(
        _training_report(
            dataset=dataset,
            feature_schema=feature_schema,
            metrics=metrics,
            selected_model_name=selected_model_name,
            model_path=model_path,
        ),
        encoding="utf-8",
    )

    return {
        "model_path": str(model_path),
        "feature_schema_path": str(feature_schema_path),
        "metrics_path": str(metrics_path),
        "training_report_path": str(training_report_path),
        "selected_model": selected_model_name,
        "metrics": metrics,
        "feature_schema": feature_schema,
        "dataset": split_summary,
    }


def _planner_metrics(split: PlannerRankerSplit, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(split.y, dtype=int)
    metrics = evaluate_predictions(labels, scores, threshold, rows=split.rows)
    predicted = (scores >= threshold).astype(int)
    utility = np.asarray(split.utility_labels, dtype=int)
    positive = labels == 1
    unsafe = utility == 0
    safe = utility == 1
    metrics["safe_priority_recall"] = _safe_div(float(np.sum(positive & (predicted == 1))), float(np.sum(positive)))
    metrics["unsafe_rejection_rate"] = _safe_div(float(np.sum(unsafe & (predicted == 0))), float(np.sum(unsafe)))
    metrics["safe_not_prioritized_count"] = int(np.sum(safe & (predicted == 0)))
    metrics["unsafe_prioritized_count"] = int(np.sum(unsafe & (predicted == 1)))
    metrics["ranking"] = _ranking_metrics(split, scores)
    return metrics


def _ranking_metrics(split: PlannerRankerSplit, scores: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-scores)
    labels = np.asarray(split.y, dtype=int)
    utility = np.asarray(split.utility_labels, dtype=int)
    top_k_values = (1, 3, 5)
    result: dict[str, Any] = {
        "priority_roc_auc": metrics_roc_auc(labels, scores),
    }
    for k in top_k_values:
        if len(order) == 0:
            result[f"top_{k}_priority_rate"] = 0.0
            result[f"top_{k}_unsafe_count"] = 0
            continue
        chosen = order[: min(k, len(order))]
        result[f"top_{k}_priority_rate"] = float(np.mean(labels[chosen])) if len(chosen) else 0.0
        result[f"top_{k}_unsafe_count"] = int(np.sum(utility[chosen] == 0))
    return result


def metrics_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    wins = 0.0
    total = float(len(positives) * len(negatives))
    for positive_score in positives:
        wins += float(np.sum(positive_score > negatives))
        wins += 0.5 * float(np.sum(positive_score == negatives))
    return float(wins / total)


def _select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = sorted({0.0, 1.0, *[float(score) for score in scores]})
    expanded: list[float] = []
    for left, right in zip(candidates, candidates[1:]):
        expanded.append(left)
        expanded.append((left + right) / 2.0)
    expanded.append(candidates[-1])
    best_threshold = 0.5
    best_key: tuple[float, float, float] | None = None
    for threshold in expanded:
        metrics = evaluate_predictions(labels.astype(int), scores, threshold)
        key = (
            float(metrics["balanced_accuracy"]) + float(metrics["true_class"]["recall"]),
            float(metrics["balanced_accuracy"]),
            float(metrics["macro_f1"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _selection_score(metrics: Mapping[str, Any]) -> float:
    return (
        float(metrics["balanced_accuracy"])
        + float(metrics["true_class"]["recall"])
        + float(metrics.get("unsafe_rejection_rate", 0.0))
    )


def _priority_target(
    *,
    utility_label: int,
    estimated_cache_gain: float,
    cache_gain_priority_threshold: float,
) -> int:
    return int(utility_label == 1 and estimated_cache_gain >= cache_gain_priority_threshold)


def _cache_gain(row: Mapping[str, Any]) -> float:
    cache_gain_report = row.get("cache_gain_report") if isinstance(row.get("cache_gain_report"), Mapping) else {}
    return _as_float(row.get("estimated_cache_gain") or cache_gain_report.get("estimated_cache_gain"))


def _split_summary(dataset: PlannerRankerDataset) -> dict[str, Any]:
    splits = {}
    for split, loaded in dataset.splits.items():
        splits[split] = {
            "sample_count": len(loaded.rows),
            "target_distribution": dict(sorted(Counter(str(bool(value)) for value in loaded.y).items())),
            "utility_distribution": dict(sorted(Counter(str(bool(value)) for value in loaded.utility_labels).items())),
            "source_distribution": dict(sorted(Counter(row.get("dataset_source") for row in loaded.rows).items())),
        }
    return {
        "splits": splits,
        "trainable_count": dataset.trainable_count,
        "target_distribution": dict(dataset.target_distribution),
        "source_distribution": dict(dataset.source_distribution),
        "cache_gain_priority_threshold": dataset.cache_gain_priority_threshold,
    }


def _training_report(
    *,
    dataset: PlannerRankerDataset,
    feature_schema: Mapping[str, Any],
    metrics: Mapping[str, Any],
    selected_model_name: str,
    model_path: Path,
) -> str:
    selected_valid = metrics["selected_valid_metrics"]
    selected_test = metrics["selected_test_metrics"]
    split_lines = []
    for split, loaded in dataset.splits.items():
        split_lines.append(
            f"- {split}: samples={len(loaded.rows)}, "
            f"target_distribution={dict(sorted(Counter(str(bool(value)) for value in loaded.y).items()))}, "
            f"utility_distribution={dict(sorted(Counter(str(bool(value)) for value in loaded.utility_labels).items()))}"
        )
    return "\n".join(
        [
            "# Planner Ranker baseline_v0 Training Report",
            "",
            "baseline_v0 is a first structured-feature planner/ranking baseline. It is not a final planner policy and is intended for replay and controlled shadow validation.",
            "",
            "## Data",
            *split_lines,
            f"- trainable samples: {dataset.trainable_count}",
            f"- cache gain priority threshold: {dataset.cache_gain_priority_threshold}",
            "",
            "## Target",
            f"- {metrics['target_definition']}",
            "- utility labels are used only to build the target, never as inference inputs",
            "- prompt text used: false",
            "",
            "## Input Features",
            f"- categorical: {', '.join(feature_schema['raw_input_fields']['categorical'])}",
            f"- multi-value: {', '.join(feature_schema['raw_input_fields']['multi_value'])}",
            f"- numeric/static placement features: {', '.join(feature_schema['raw_input_fields']['numeric'])}",
            "",
            "## Excluded Leakage Fields",
            ", ".join(LEAKAGE_FIELDS),
            "",
            "## Models",
            "- Logistic Regression with balanced weighted logistic loss.",
            "- Random Forest with balanced bootstrap samples.",
            f"- final selected model by validation score: {selected_model_name}",
            "",
            "## Validation Metrics",
            f"- balanced_accuracy: {selected_valid['balanced_accuracy']:.3f}",
            f"- priority precision/recall/f1: {selected_valid['true_class']['precision']:.3f} / {selected_valid['true_class']['recall']:.3f} / {selected_valid['true_class']['f1']:.3f}",
            f"- unsafe_rejection_rate: {selected_valid['unsafe_rejection_rate']:.3f}",
            f"- unsafe_prioritized_count: {selected_valid['unsafe_prioritized_count']}",
            f"- top_3_priority_rate: {selected_valid['ranking']['top_3_priority_rate']:.3f}",
            "",
            "## Test Metrics",
            f"- balanced_accuracy: {selected_test['balanced_accuracy']:.3f}",
            f"- priority precision/recall/f1: {selected_test['true_class']['precision']:.3f} / {selected_test['true_class']['recall']:.3f} / {selected_test['true_class']['f1']:.3f}",
            f"- unsafe_rejection_rate: {selected_test['unsafe_rejection_rate']:.3f}",
            f"- unsafe_prioritized_count: {selected_test['unsafe_prioritized_count']}",
            f"- top_3_priority_rate: {selected_test['ranking']['top_3_priority_rate']:.3f}",
            "",
            "## Artifacts",
            f"- model: {model_path}",
            "- feature_schema.json",
            "- metrics.json",
            "- training_report.md",
            "",
        ]
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        if isinstance(value, bool):
            return float(int(value))
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Planner Ranker baseline_v0 from expanded prompt-safe features.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--dataset-root", default="datasets/utility_validator")
    parser.add_argument("--output-dir", default="artifacts/planner_ranker/baseline_v0")
    parser.add_argument("--logistic-iterations", type=int, default=3000)
    parser.add_argument("--random-forest-estimators", type=int, default=80)
    args = parser.parse_args(argv)
    result = train_planner_ranker_baseline_v0(
        repo_root=args.repo_root,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        logistic_iterations=args.logistic_iterations,
        random_forest_estimators=args.random_forest_estimators,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "metrics"}, indent=2, ensure_ascii=False))
    return 0


def _register_pickle_module_alias() -> None:
    module_name = __spec__.name if __spec__ is not None and __spec__.name else None
    if __name__ != "__main__" or not module_name:
        return
    sys.modules[module_name] = sys.modules[__name__]
    for cls in (
        PlannerRankerModel,
        PlannerRankerSplit,
        PlannerRankerDataset,
    ):
        cls.__module__ = module_name


if __name__ == "__main__":
    _register_pickle_module_alias()
    raise SystemExit(main())
