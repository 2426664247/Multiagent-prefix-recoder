from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXPANDED_SPLITS = ("expanded_train", "expanded_valid", "expanded_test")
SPLIT_ALIAS_FILE = {
    "expanded_train": "train_features.jsonl",
    "expanded_valid": "valid_features.jsonl",
    "expanded_test": "test_features.jsonl",
}

TARGET_FIELD = "is_utility_preserved"
REQUIRED_LABEL_SOURCE = "dsapi_execution_oracle"

LEAKAGE_FIELDS = (
    "is_utility_preserved",
    "failure_type",
    "original_run_status",
    "reordered_run_status",
    "label_source",
    "api_model",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "cached_tokens_delta",
    "latency",
    "latency_seconds",
    "estimated_cost",
    "estimated_cost_usd",
    "whether_label_is_utility_verified",
    "api_cost_estimate",
    "api_run_reports",
    "oracle_reports",
    "validator_report",
    "utility_status",
)

PROMPT_TEXT_FIELDS = (
    "prompt_text",
    "original_prompt_text",
    "reordered_prompt_text",
    "source_prompt",
    "candidate_text",
    "original_messages",
    "rewritten_messages",
    "messages",
)

CATEGORICAL_FIELDS = ("dataset_source", "scenario_type", "candidate_strategy")
MULTI_VALUE_FIELDS = ("source_scopes", "target_scopes", "risk_tags", "dependency_notes")

NUMERIC_FEATURES = (
    "moved_block_count",
    "global_prefix_tokens_or_chars",
    "subgroup_prefix_tokens_or_chars",
    "estimated_cache_gain",
    "hard_warning_count",
    "movement_distance_summary.max_abs_distance",
    "movement_distance_summary.moved_position_count",
    "movement_distance_summary.sum_abs_distance",
    "placement_changes.count",
    "placement_changes.moved_count",
    "placement_changes.global_target_count",
    "placement_changes.subgroup_target_count",
    "placement_changes.agent_local_target_count",
    "placement_changes.cache_contribution_sum",
    "placement_changes.placement_score_avg",
    "placement_changes.placement_score_max",
    "placement_changes.dependency_note_count",
    "placement_changes.risk_tag_count",
    "cache_gain_report.longest_common_prefix_tokens",
    "cache_gain_report.global_prefix_tokens",
    "cache_gain_report.subgroup_prefix_tokens",
    "cache_gain_report.estimated_cache_gain",
    "cache_gain_report.node_count",
    "cache_gain_report.node_token_len_sum",
    "cache_gain_report.node_cache_contribution_sum",
    "cache_gain_report.placement_cache_contribution_sum",
)


@dataclass(frozen=True)
class SplitPaths:
    split: str
    feature_path: Path
    label_path: Path


@dataclass(frozen=True)
class LoadedSplit:
    split: str
    feature_path: Path
    label_path: Path
    rows: tuple[dict[str, Any], ...]
    y: tuple[int, ...]
    sample_ids: tuple[str, ...]
    label_count: int
    skipped_label_count: int
    label_distribution: Mapping[str, int]
    feature_distribution: Mapping[str, int]


@dataclass(frozen=True)
class LoadedDataset:
    splits: Mapping[str, LoadedSplit]
    skipped_label_count: int
    trainable_count: int
    label_distribution: Mapping[str, int]
    source_distribution: Mapping[str, int]


@dataclass
class UtilityValidatorModel:
    encoder: "StructuredFeatureEncoder"
    estimator: Any
    threshold: float
    model_name: str
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "UtilityValidatorModel":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"Expected UtilityValidatorModel pickle, got {type(model).__name__}")
        return model

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    def predict_proba(self, row: Mapping[str, Any]) -> float:
        matrix = self.encoder.transform([dict(row)])
        return float(self.estimator.predict_proba(matrix)[0])

    def predict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        probability = self.predict_proba(row)
        predicted = probability >= self.threshold
        return {
            "is_utility_preserved": bool(predicted),
            "confidence": float(probability if predicted else 1.0 - probability),
            "failure_type": None,
            "reason": "baseline model prediction from structured placement features",
            "model_name": self.model_name,
        }

    def evaluate(self, **kwargs: Any) -> Any:
        from .validator import UtilityPreservationReport

        row = runtime_feature_row_from_gate_inputs(**kwargs)
        prediction = self.predict(row)
        status = "passed" if prediction["is_utility_preserved"] else "failed"
        return UtilityPreservationReport(
            is_utility_preserved=prediction["is_utility_preserved"],
            confidence=prediction["confidence"],
            reason=prediction["reason"],
            checks=("structured_placement_features", "baseline_v0"),
            model_name=self.model_name,
            prompt_safe=True,
            utility_status=status,
        )


class StructuredFeatureEncoder:
    def __init__(
        self,
        *,
        categorical_values: Mapping[str, tuple[str, ...]],
        multi_values: Mapping[str, tuple[str, ...]],
        numeric_means: Mapping[str, float],
        numeric_stds: Mapping[str, float],
    ) -> None:
        self.categorical_values = {key: tuple(values) for key, values in categorical_values.items()}
        self.multi_values = {key: tuple(values) for key, values in multi_values.items()}
        self.numeric_means = dict(numeric_means)
        self.numeric_stds = dict(numeric_stds)
        self.numeric_features = NUMERIC_FEATURES
        self.feature_names = self._feature_names()

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, Any]]) -> "StructuredFeatureEncoder":
        categorical_values: dict[str, tuple[str, ...]] = {}
        for field in CATEGORICAL_FIELDS:
            categorical_values[field] = tuple(sorted({_as_string(row.get(field)) for row in rows if row.get(field) is not None}))

        multi_values: dict[str, tuple[str, ...]] = {}
        for field in MULTI_VALUE_FIELDS:
            values: set[str] = set()
            for row in rows:
                values.update(_as_string(item) for item in _as_sequence(row.get(field)))
            multi_values[field] = tuple(sorted(values))

        raw_numeric = np.asarray([_numeric_values(row) for row in rows], dtype=float)
        means = raw_numeric.mean(axis=0) if len(raw_numeric) else np.zeros(len(NUMERIC_FEATURES), dtype=float)
        stds = raw_numeric.std(axis=0) if len(raw_numeric) else np.ones(len(NUMERIC_FEATURES), dtype=float)
        stds = np.where(stds < 1e-9, 1.0, stds)
        return cls(
            categorical_values=categorical_values,
            multi_values=multi_values,
            numeric_means={name: float(value) for name, value in zip(NUMERIC_FEATURES, means)},
            numeric_stds={name: float(value) for name, value in zip(NUMERIC_FEATURES, stds)},
        )

    def transform(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        encoded_rows: list[list[float]] = []
        for row in rows:
            values: list[float] = []
            numeric = _numeric_values(row)
            for name, value in zip(NUMERIC_FEATURES, numeric):
                mean = self.numeric_means.get(name, 0.0)
                std = self.numeric_stds.get(name, 1.0) or 1.0
                values.append((float(value) - mean) / std)

            for field in CATEGORICAL_FIELDS:
                raw = _as_string(row.get(field))
                categories = self.categorical_values.get(field, ())
                values.extend(1.0 if raw == category else 0.0 for category in categories)
                values.append(1.0 if raw and raw not in categories else 0.0)

            for field in MULTI_VALUE_FIELDS:
                raw_values = set(_as_string(item) for item in _as_sequence(row.get(field)))
                categories = self.multi_values.get(field, ())
                values.extend(1.0 if category in raw_values else 0.0 for category in categories)
                unknown_count = len(raw_values.difference(categories))
                values.append(float(unknown_count))

            encoded_rows.append(values)
        return np.asarray(encoded_rows, dtype=float)

    def to_schema(self) -> dict[str, Any]:
        return {
            "schema_version": "utility-validator-baseline-feature-schema-v0",
            "target_field": TARGET_FIELD,
            "raw_input_fields": {
                "categorical": list(CATEGORICAL_FIELDS),
                "multi_value": list(MULTI_VALUE_FIELDS),
                "numeric": list(NUMERIC_FEATURES),
            },
            "categorical_values": {key: list(values) for key, values in self.categorical_values.items()},
            "multi_value_categories": {key: list(values) for key, values in self.multi_values.items()},
            "numeric_means": self.numeric_means,
            "numeric_stds": self.numeric_stds,
            "encoded_feature_names": list(self.feature_names),
            "excluded_leakage_fields": list(LEAKAGE_FIELDS),
            "prompt_text_used": False,
        }

    def _feature_names(self) -> tuple[str, ...]:
        names: list[str] = list(NUMERIC_FEATURES)
        for field in CATEGORICAL_FIELDS:
            names.extend(f"{field}={category}" for category in self.categorical_values.get(field, ()))
            names.append(f"{field}=__unknown__")
        for field in MULTI_VALUE_FIELDS:
            names.extend(f"{field} contains {category}" for category in self.multi_values.get(field, ()))
            names.append(f"{field} unknown_count")
        return tuple(names)


class LogisticRegressionBaseline:
    def __init__(
        self,
        *,
        learning_rate: float = 0.08,
        iterations: int = 3000,
        l2: float = 0.01,
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2 = l2
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, matrix: np.ndarray, labels: np.ndarray) -> "LogisticRegressionBaseline":
        y = labels.astype(float)
        n_samples, n_features = matrix.shape
        positive_count = max(1, int(y.sum()))
        negative_count = max(1, int(n_samples - y.sum()))
        positive_weight = n_samples / (2.0 * positive_count)
        negative_weight = n_samples / (2.0 * negative_count)
        weights = np.where(y == 1.0, positive_weight, negative_weight)
        weight_sum = float(weights.sum()) or 1.0
        coef = np.zeros(n_features, dtype=float)
        intercept = 0.0

        for _ in range(self.iterations):
            logits = np.clip(matrix @ coef + intercept, -35.0, 35.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            error = weights * (probs - y)
            grad = (matrix.T @ error) / weight_sum + self.l2 * coef
            intercept_grad = float(error.sum() / weight_sum)
            coef -= self.learning_rate * grad
            intercept -= self.learning_rate * intercept_grad

        self.coef_ = coef
        self.intercept_ = float(intercept)
        return self

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model is not fitted")
        logits = np.clip(matrix @ self.coef_ + self.intercept_, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-logits))


@dataclass
class _TreeNode:
    probability: float
    feature_index: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None


class DecisionTreeBaseline:
    def __init__(
        self,
        *,
        max_depth: int = 5,
        min_samples_leaf: int = 3,
        max_features: int | None = None,
        random_state: int = 0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root_: _TreeNode | None = None

    def fit(self, matrix: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray | None = None) -> "DecisionTreeBaseline":
        weights = sample_weight if sample_weight is not None else np.ones(len(labels), dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.root_ = self._build(matrix, labels.astype(int), weights.astype(float), depth=0, rng=rng)
        return self

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("Model is not fitted")
        return np.asarray([self._predict_one(row, self.root_) for row in matrix], dtype=float)

    def _build(
        self,
        matrix: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        *,
        depth: int,
        rng: np.random.Generator,
    ) -> _TreeNode:
        probability = _weighted_positive_rate(labels, weights)
        if depth >= self.max_depth or len(labels) < self.min_samples_leaf * 2 or probability in {0.0, 1.0}:
            return _TreeNode(probability=probability)

        feature_count = matrix.shape[1]
        max_features = self.max_features or max(1, int(math.sqrt(feature_count)))
        feature_indices = rng.choice(feature_count, size=min(max_features, feature_count), replace=False)
        split = self._best_split(matrix, labels, weights, feature_indices)
        if split is None:
            return _TreeNode(probability=probability)

        feature_index, threshold = split
        mask = matrix[:, feature_index] <= threshold
        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
            return _TreeNode(probability=probability)
        return _TreeNode(
            probability=probability,
            feature_index=int(feature_index),
            threshold=float(threshold),
            left=self._build(matrix[mask], labels[mask], weights[mask], depth=depth + 1, rng=rng),
            right=self._build(matrix[~mask], labels[~mask], weights[~mask], depth=depth + 1, rng=rng),
        )

    def _best_split(
        self,
        matrix: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        feature_indices: np.ndarray,
    ) -> tuple[int, float] | None:
        best_gain = 0.0
        best: tuple[int, float] | None = None
        parent_impurity = _weighted_gini(labels, weights)
        total_weight = float(weights.sum()) or 1.0
        for feature_index in feature_indices:
            column = matrix[:, feature_index]
            unique_values = np.unique(column)
            if len(unique_values) <= 1:
                continue
            thresholds = _candidate_thresholds(unique_values)
            for threshold in thresholds:
                mask = column <= threshold
                left_count = int(mask.sum())
                right_count = int((~mask).sum())
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue
                left_weight = float(weights[mask].sum())
                right_weight = float(weights[~mask].sum())
                impurity = (
                    left_weight / total_weight * _weighted_gini(labels[mask], weights[mask])
                    + right_weight / total_weight * _weighted_gini(labels[~mask], weights[~mask])
                )
                gain = parent_impurity - impurity
                if gain > best_gain + 1e-12:
                    best_gain = float(gain)
                    best = (int(feature_index), float(threshold))
        return best

    def _predict_one(self, row: np.ndarray, node: _TreeNode) -> float:
        current = node
        while current.feature_index is not None and current.threshold is not None:
            if row[current.feature_index] <= current.threshold:
                if current.left is None:
                    break
                current = current.left
            else:
                if current.right is None:
                    break
                current = current.right
        return float(current.probability)


class RandomForestBaseline:
    def __init__(
        self,
        *,
        n_estimators: int = 80,
        max_depth: int = 5,
        min_samples_leaf: int = 3,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees: list[DecisionTreeBaseline] = []

    def fit(self, matrix: np.ndarray, labels: np.ndarray) -> "RandomForestBaseline":
        rng = np.random.default_rng(self.random_state)
        positive_indices = np.flatnonzero(labels == 1)
        negative_indices = np.flatnonzero(labels == 0)
        if len(positive_indices) == 0 or len(negative_indices) == 0:
            raise ValueError("RandomForestBaseline requires both classes")
        per_class = max(len(positive_indices), len(negative_indices))
        feature_count = matrix.shape[1]
        max_features = max(1, int(math.sqrt(feature_count)))
        self.trees = []
        for tree_index in range(self.n_estimators):
            sampled_positive = rng.choice(positive_indices, size=per_class, replace=True)
            sampled_negative = rng.choice(negative_indices, size=per_class, replace=True)
            sampled = np.concatenate([sampled_positive, sampled_negative])
            rng.shuffle(sampled)
            tree = DecisionTreeBaseline(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=self.random_state + tree_index + 1,
            )
            tree.fit(matrix[sampled], labels[sampled], sample_weight=np.ones(len(sampled), dtype=float))
            self.trees.append(tree)
        return self

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise ValueError("Model is not fitted")
        stacked = np.vstack([tree.predict_proba(matrix) for tree in self.trees])
        return stacked.mean(axis=0)


def discover_expanded_dataset_paths(
    repo_root: str | Path = ".",
    dataset_root: str | Path = "datasets/utility_validator",
) -> dict[str, SplitPaths]:
    repo = Path(repo_root)
    root = _resolve_under_repo(repo, dataset_root)
    summary_paths = _paths_from_expanded_summary(repo, root)
    split_paths: dict[str, SplitPaths] = {}
    for split in EXPANDED_SPLITS:
        feature_candidates: list[Path] = []
        label_candidates: list[Path] = []
        if split in summary_paths.get("feature_aliases", {}):
            feature_candidates.append(_resolve_under_repo(repo, summary_paths["feature_aliases"][split]))
        if split in summary_paths.get("features", {}):
            feature_candidates.append(_resolve_under_repo(repo, summary_paths["features"][split]))
        if split in summary_paths.get("labels", {}):
            label_candidates.append(_resolve_under_repo(repo, summary_paths["labels"][split]))

        feature_candidates.extend(
            [
                root / "features" / split / SPLIT_ALIAS_FILE[split],
                root / "features" / split / "training_features.jsonl",
            ]
        )
        label_candidates.append(root / "labels" / split / "utility_labels.jsonl")

        feature_path = _first_existing(feature_candidates, f"feature path for {split}")
        label_path = _first_existing(label_candidates, f"label path for {split}")
        _assert_expanded_safe_path(feature_path)
        _assert_expanded_safe_path(label_path)
        split_paths[split] = SplitPaths(split=split, feature_path=feature_path, label_path=label_path)
    return split_paths


def load_expanded_utility_validator_dataset(
    repo_root: str | Path = ".",
    dataset_root: str | Path = "datasets/utility_validator",
) -> LoadedDataset:
    split_paths = discover_expanded_dataset_paths(repo_root=repo_root, dataset_root=dataset_root)
    loaded: dict[str, LoadedSplit] = {}
    all_label_distribution: Counter[str] = Counter()
    source_distribution: Counter[str] = Counter()
    skipped_label_count = 0
    trainable_count = 0
    for split, paths in split_paths.items():
        labels = _read_jsonl(paths.label_path)
        features = _read_jsonl(paths.feature_path)
        _validate_prompt_safe_rows(labels, paths.label_path)
        _validate_prompt_safe_rows(features, paths.feature_path)
        if any(label.get("label_source") == "fake_smoke_oracle" for label in labels):
            raise ValueError(f"fake_smoke_oracle label found in expanded split {split}")
        if any(label.get("label_source") != REQUIRED_LABEL_SOURCE for label in labels):
            raise ValueError(f"non-{REQUIRED_LABEL_SOURCE} label found in expanded split {split}")

        label_by_id = {str(label.get("label_id")): label for label in labels}
        rows: list[dict[str, Any]] = []
        y: list[int] = []
        sample_ids: list[str] = []
        feature_distribution: Counter[str] = Counter()
        for feature in features:
            label_id = str(feature.get("label_id"))
            label = label_by_id.get(label_id)
            if label is None:
                raise ValueError(f"Feature row references missing label_id {label_id!r} in {paths.feature_path}")
            target = label.get(TARGET_FIELD)
            if target not in {True, False}:
                continue
            rows.append(dict(feature))
            y.append(1 if target is True else 0)
            sample_ids.append(str(feature.get("sample_id") or label_id))
            feature_distribution[str(target)] += 1
            source_distribution[str(feature.get("dataset_source"))] += 1

        label_distribution = Counter(str(label.get(TARGET_FIELD)) for label in labels)
        all_label_distribution.update(label_distribution)
        split_skipped = sum(1 for label in labels if label.get(TARGET_FIELD) not in {True, False})
        skipped_label_count += split_skipped
        trainable_count += len(rows)
        loaded[split] = LoadedSplit(
            split=split,
            feature_path=paths.feature_path,
            label_path=paths.label_path,
            rows=tuple(rows),
            y=tuple(y),
            sample_ids=tuple(sample_ids),
            label_count=len(labels),
            skipped_label_count=split_skipped,
            label_distribution=dict(sorted(label_distribution.items())),
            feature_distribution=dict(sorted(feature_distribution.items())),
        )

    return LoadedDataset(
        splits=loaded,
        skipped_label_count=skipped_label_count,
        trainable_count=trainable_count,
        label_distribution=dict(sorted(all_label_distribution.items())),
        source_distribution=dict(sorted(source_distribution.items())),
    )


def train_utility_validator_baseline_v0(
    *,
    repo_root: str | Path = ".",
    dataset_root: str | Path = "datasets/utility_validator",
    output_dir: str | Path = "artifacts/utility_validator/baseline_v0",
    random_seed: int = 42,
    logistic_iterations: int = 3000,
    random_forest_estimators: int = 80,
) -> dict[str, Any]:
    dataset = load_expanded_utility_validator_dataset(repo_root=repo_root, dataset_root=dataset_root)
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
            "train": evaluate_predictions(y_train, estimator.predict_proba(x_train), threshold, rows=train.rows),
            "valid": evaluate_predictions(y_valid, valid_scores, threshold, rows=valid.rows),
            "test": evaluate_predictions(y_test, estimator.predict_proba(x_test), threshold, rows=test.rows),
        }

    selected_model_name = sorted(
        model_metrics,
        key=lambda item: (
            model_metrics[item]["selection_score"],
            model_metrics[item]["valid"]["negative_class"]["recall"],
            model_metrics[item]["valid"]["balanced_accuracy"],
            item,
        ),
        reverse=True,
    )[0]

    selected_model = UtilityValidatorModel(
        encoder=encoder,
        estimator=baselines[selected_model_name],
        threshold=thresholds[selected_model_name],
        model_name=f"utility_validator_baseline_v0:{selected_model_name}",
        metadata={
            "schema_version": "utility-validator-baseline-v0",
            "selected_model": selected_model_name,
            "threshold": thresholds[selected_model_name],
            "trained_at_note": "baseline_v0 trained from expanded prompt-safe structured features",
            "prompt_text_used": False,
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
    _write_json(feature_schema_path, feature_schema)

    split_summary = _split_summary(dataset)
    metrics = {
        "schema_version": "utility-validator-baseline-v0-metrics",
        "dataset": split_summary,
        "models": model_metrics,
        "selected_model": selected_model_name,
        "selected_threshold": thresholds[selected_model_name],
        "selected_valid_metrics": model_metrics[selected_model_name]["valid"],
        "selected_test_metrics": model_metrics[selected_model_name]["test"],
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


def evaluate_predictions(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    y = labels.astype(int)
    predicted = (scores >= threshold).astype(int)
    actual_false = y == 0
    actual_true = y == 1
    predicted_false = predicted == 0
    predicted_true = predicted == 1

    false_as_false = int(np.sum(actual_false & predicted_false))
    false_as_true = int(np.sum(actual_false & predicted_true))
    true_as_false = int(np.sum(actual_true & predicted_false))
    true_as_true = int(np.sum(actual_true & predicted_true))

    true_metrics = _class_metrics(true_as_true, false_as_true, true_as_false)
    false_metrics = _class_metrics(false_as_false, true_as_false, false_as_true)
    accuracy = _safe_div(false_as_false + true_as_true, len(y))
    balanced_accuracy = (true_metrics["recall"] + false_metrics["recall"]) / 2.0
    macro_f1 = (true_metrics["f1"] + false_metrics["f1"]) / 2.0
    result = {
        "sample_count": int(len(y)),
        "threshold": float(threshold),
        "class_distribution": {
            "false": int(np.sum(actual_false)),
            "true": int(np.sum(actual_true)),
        },
        "confusion_matrix": {
            "labels": [False, True],
            "matrix": [
                [false_as_false, false_as_true],
                [true_as_false, true_as_true],
            ],
        },
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "true_class": true_metrics,
        "negative_class": false_metrics,
        "negative_recall": false_metrics["recall"],
        "false_negative_rate": _safe_div(false_as_true, false_as_false + false_as_true),
        "roc_auc": _roc_auc(y, scores),
        "negative_pr_auc": _pr_auc(1 - y, 1.0 - scores),
    }
    if rows is not None:
        result["per_source"] = _per_source_metrics(rows, y, scores, threshold)
    return result


def runtime_feature_row_from_gate_inputs(**kwargs: Any) -> dict[str, Any]:
    compile_result = kwargs.get("compile_result")
    plan = kwargs.get("plan")
    cache_utility_estimate = kwargs.get("cache_utility_estimate")
    placements = tuple(getattr(plan, "placements", ()) or ())
    original_order = tuple(getattr(plan, "original_order", ()) or ())
    new_order = tuple(getattr(plan, "new_order", ()) or ())
    original_index = {block_id: index for index, block_id in enumerate(original_order)}
    new_index = {block_id: index for index, block_id in enumerate(new_order)}
    distances = [
        abs(original_index[block_id] - new_index[block_id])
        for block_id in original_order
        if block_id in new_index and original_index[block_id] != new_index[block_id]
    ]
    cache_report = dict(getattr(plan, "cache_gain_report", None) or {})
    candidate = getattr(plan, "prefix_tree_candidate", None)
    strategy = "runtime"
    if candidate is not None:
        reason = str(getattr(candidate, "generation_reason", "") or "")
        if ":" in reason:
            strategy = reason.split(":", 1)[0]
        elif reason:
            strategy = reason
    row = {
        "dataset_source": "runtime",
        "scenario_type": "utility_gate_replay",
        "candidate_strategy": strategy,
        "source_scopes": [_scope_value(getattr(placement, "original_scope", None)) for placement in placements],
        "target_scopes": [_scope_value(getattr(placement, "target_scope", None)) for placement in placements],
        "risk_tags": sorted({tag for placement in placements for tag in getattr(placement, "risk_tags", ())}),
        "dependency_notes": sorted({note for placement in placements for note in getattr(placement, "dependency_notes", ())}),
        "moved_block_count": len(tuple(getattr(plan, "moved_blocks", ()) or ())),
        "movement_distance_summary": {
            "max_abs_distance": max(distances) if distances else 0,
            "moved_position_count": len(distances),
            "sum_abs_distance": sum(distances),
        },
        "global_prefix_tokens_or_chars": int(
            getattr(cache_utility_estimate, "global_prefix_tokens", None)
            or cache_report.get("global_prefix_tokens")
            or 0
        ),
        "subgroup_prefix_tokens_or_chars": int(
            getattr(cache_utility_estimate, "subgroup_prefix_tokens", None)
            or cache_report.get("subgroup_prefix_tokens")
            or 0
        ),
        "estimated_cache_gain": float(
            getattr(cache_utility_estimate, "estimated_gain_chars", None)
            or getattr(plan, "estimated_cache_gain", None)
            or cache_report.get("estimated_cache_gain")
            or 0.0
        ),
        "hard_warning_count": len(tuple(getattr(plan, "risk_notes", ()) or ())),
        "placement_changes": [_placement_to_feature_dict(placement) for placement in placements],
        "cache_gain_report": cache_report,
    }
    if compile_result is not None:
        row["scenario_type"] = getattr(compile_result, "scenario_type", None) or row["scenario_type"]
    return row


def _numeric_values(row: Mapping[str, Any]) -> list[float]:
    movement = row.get("movement_distance_summary") if isinstance(row.get("movement_distance_summary"), Mapping) else {}
    placement = _placement_change_features(row.get("placement_changes"))
    cache_report = _cache_gain_report_features(row.get("cache_gain_report"))
    return [
        _as_float(row.get("moved_block_count")),
        _as_float(row.get("global_prefix_tokens_or_chars")),
        _as_float(row.get("subgroup_prefix_tokens_or_chars")),
        _as_float(row.get("estimated_cache_gain")),
        _as_float(row.get("hard_warning_count")),
        _as_float(movement.get("max_abs_distance")),
        _as_float(movement.get("moved_position_count")),
        _as_float(movement.get("sum_abs_distance")),
        placement["count"],
        placement["moved_count"],
        placement["global_target_count"],
        placement["subgroup_target_count"],
        placement["agent_local_target_count"],
        placement["cache_contribution_sum"],
        placement["placement_score_avg"],
        placement["placement_score_max"],
        placement["dependency_note_count"],
        placement["risk_tag_count"],
        cache_report["longest_common_prefix_tokens"],
        cache_report["global_prefix_tokens"],
        cache_report["subgroup_prefix_tokens"],
        cache_report["estimated_cache_gain"],
        cache_report["node_count"],
        cache_report["node_token_len_sum"],
        cache_report["node_cache_contribution_sum"],
        cache_report["placement_cache_contribution_sum"],
    ]


def _placement_change_features(value: Any) -> dict[str, float]:
    placements = list(value) if isinstance(value, list) else []
    scores: list[float] = []
    result = {
        "count": float(len(placements)),
        "moved_count": 0.0,
        "global_target_count": 0.0,
        "subgroup_target_count": 0.0,
        "agent_local_target_count": 0.0,
        "cache_contribution_sum": 0.0,
        "placement_score_avg": 0.0,
        "placement_score_max": 0.0,
        "dependency_note_count": 0.0,
        "risk_tag_count": 0.0,
    }
    for placement in placements:
        if not isinstance(placement, Mapping):
            continue
        if placement.get("moved") is True:
            result["moved_count"] += 1.0
        target_scope = _scope_value(placement.get("target_scope"))
        if target_scope == "global":
            result["global_target_count"] += 1.0
        elif target_scope == "subgroup":
            result["subgroup_target_count"] += 1.0
        elif target_scope == "agent_local":
            result["agent_local_target_count"] += 1.0
        result["cache_contribution_sum"] += _as_float(placement.get("cache_contribution"))
        score = _as_float(placement.get("placement_score"))
        scores.append(score)
        result["dependency_note_count"] += float(len(_as_sequence(placement.get("dependency_notes"))))
        result["risk_tag_count"] += float(len(_as_sequence(placement.get("risk_tags"))))
    if scores:
        result["placement_score_avg"] = float(sum(scores) / len(scores))
        result["placement_score_max"] = float(max(scores))
    return result


def _cache_gain_report_features(value: Any) -> dict[str, float]:
    report = value if isinstance(value, Mapping) else {}
    node_contributions = report.get("node_cache_contributions")
    placement_contributions = report.get("placement_cache_contributions")
    node_values = list(node_contributions.values()) if isinstance(node_contributions, Mapping) else []
    return {
        "longest_common_prefix_tokens": _as_float(report.get("longest_common_prefix_tokens")),
        "global_prefix_tokens": _as_float(report.get("global_prefix_tokens")),
        "subgroup_prefix_tokens": _as_float(report.get("subgroup_prefix_tokens")),
        "estimated_cache_gain": _as_float(report.get("estimated_cache_gain")),
        "node_count": float(len(node_values)),
        "node_token_len_sum": float(
            sum(_as_float(node.get("token_len")) for node in node_values if isinstance(node, Mapping))
        ),
        "node_cache_contribution_sum": float(
            sum(_as_float(node.get("cache_contribution")) for node in node_values if isinstance(node, Mapping))
        ),
        "placement_cache_contribution_sum": float(
            sum(_as_float(item) for item in placement_contributions.values())
            if isinstance(placement_contributions, Mapping)
            else 0.0
        ),
    }


def _select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = sorted({0.0, 1.0, *[float(score) for score in scores]})
    expanded: list[float] = []
    for left, right in zip(candidates, candidates[1:]):
        expanded.append(left)
        expanded.append((left + right) / 2.0)
    expanded.append(candidates[-1])
    best_threshold = 0.5
    best_key: tuple[float, float, float, float] | None = None
    for threshold in expanded:
        metrics = evaluate_predictions(labels, scores, threshold)
        key = (
            float(metrics["balanced_accuracy"]) + float(metrics["negative_class"]["recall"]),
            float(metrics["balanced_accuracy"]),
            float(metrics["negative_class"]["recall"]),
            float(metrics["macro_f1"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _selection_score(metrics: Mapping[str, Any]) -> float:
    return float(metrics["balanced_accuracy"]) + float(metrics["negative_class"]["recall"])


def _per_source_metrics(
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    indexes_by_source: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indexes_by_source[_as_string(row.get("dataset_source"))].append(index)
    result: dict[str, Any] = {}
    for source, indexes in sorted(indexes_by_source.items()):
        source_labels = labels[indexes]
        source_scores = scores[indexes]
        source_metrics = evaluate_predictions(source_labels, source_scores, threshold)
        source_metrics.pop("per_source", None)
        result[source] = source_metrics
    return result


def _class_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": int(tp + fn),
    }


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
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


def _pr_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if int(labels.sum()) == 0:
        return None
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = 0.0
    fp = 0.0
    points: list[tuple[float, float]] = [(0.0, 1.0)]
    positives = float(labels.sum())
    for label in sorted_labels:
        if label == 1:
            tp += 1.0
        else:
            fp += 1.0
        recall = tp / positives
        precision = tp / (tp + fp)
        points.append((recall, precision))
    area = 0.0
    previous_recall, previous_precision = points[0]
    for recall, precision in points[1:]:
        area += (recall - previous_recall) * ((precision + previous_precision) / 2.0)
        previous_recall = recall
        previous_precision = precision
    return float(area)


def _weighted_positive_rate(labels: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float(weights[labels == 1].sum() / total)


def _weighted_gini(labels: np.ndarray, weights: np.ndarray) -> float:
    positive_rate = _weighted_positive_rate(labels, weights)
    negative_rate = 1.0 - positive_rate
    return 1.0 - positive_rate * positive_rate - negative_rate * negative_rate


def _candidate_thresholds(unique_values: np.ndarray) -> np.ndarray:
    if len(unique_values) <= 32:
        return (unique_values[:-1] + unique_values[1:]) / 2.0
    quantiles = np.linspace(0.05, 0.95, num=24)
    return np.unique(np.quantile(unique_values, quantiles))


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _split_summary(dataset: LoadedDataset) -> dict[str, Any]:
    splits = {}
    for split, loaded in dataset.splits.items():
        splits[split] = {
            "feature_path": str(loaded.feature_path),
            "label_path": str(loaded.label_path),
            "sample_count": len(loaded.rows),
            "label_count": loaded.label_count,
            "skipped_label_count": loaded.skipped_label_count,
            "label_distribution": dict(loaded.label_distribution),
            "feature_distribution": dict(loaded.feature_distribution),
            "source_distribution": dict(sorted(Counter(row.get("dataset_source") for row in loaded.rows).items())),
        }
    return {
        "splits": splits,
        "trainable_count": dataset.trainable_count,
        "skipped_label_count": dataset.skipped_label_count,
        "label_distribution": dict(dataset.label_distribution),
        "source_distribution": dict(dataset.source_distribution),
    }


def _training_report(
    *,
    dataset: LoadedDataset,
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
            f"- {split}: features={len(loaded.rows)}, labels={loaded.label_count}, "
            f"skipped={loaded.skipped_label_count}, distribution={dict(loaded.feature_distribution)}, "
            f"feature_path={loaded.feature_path}, label_path={loaded.label_path}"
        )
    per_source = selected_test.get("per_source", {})
    per_source_lines = [
        f"- {source}: n={item['sample_count']}, balanced_accuracy={item['balanced_accuracy']:.3f}, "
        f"negative_recall={item['negative_class']['recall']:.3f}, false_negative_rate={item['false_negative_rate']:.3f}"
        for source, item in per_source.items()
    ]
    return "\n".join(
        [
            "# Utility Validator baseline_v0 Training Report",
            "",
            "baseline_v0 is a first structured-feature baseline. It is not the final Utility Validator; replay validation and main-flow Utility Preservation Gate integration still need follow-up verification.",
            "",
            "## Data",
            *split_lines,
            f"- trainable samples: {dataset.trainable_count}",
            f"- skipped labels excluded from training: {dataset.skipped_label_count}",
            f"- overall label distribution: {dict(dataset.label_distribution)}",
            "",
            "## Input Features",
            f"- categorical: {', '.join(feature_schema['raw_input_fields']['categorical'])}",
            f"- multi-value: {', '.join(feature_schema['raw_input_fields']['multi_value'])}",
            f"- numeric/static placement features: {', '.join(feature_schema['raw_input_fields']['numeric'])}",
            "- prompt text used: false",
            "",
            "## Excluded Leakage Fields",
            ", ".join(LEAKAGE_FIELDS),
            "",
            "## Models",
            "- Logistic Regression with class_weight=\"balanced\" implemented as weighted logistic loss.",
            "- Random Forest with balanced bootstrap samples.",
            f"- final selected model by validation score: {selected_model_name}",
            "",
            "## Validation Metrics",
            f"- balanced_accuracy: {selected_valid['balanced_accuracy']:.3f}",
            f"- true precision/recall/f1: {selected_valid['true_class']['precision']:.3f} / {selected_valid['true_class']['recall']:.3f} / {selected_valid['true_class']['f1']:.3f}",
            f"- false precision/recall/f1: {selected_valid['negative_class']['precision']:.3f} / {selected_valid['negative_class']['recall']:.3f} / {selected_valid['negative_class']['f1']:.3f}",
            f"- negative_recall: {selected_valid['negative_class']['recall']:.3f}",
            f"- false_negative_rate: {selected_valid['false_negative_rate']:.3f}",
            f"- macro_f1: {selected_valid['macro_f1']:.3f}",
            f"- negative_pr_auc: {selected_valid['negative_pr_auc']}",
            f"- roc_auc: {selected_valid['roc_auc']}",
            f"- confusion_matrix labels [false, true]: {selected_valid['confusion_matrix']['matrix']}",
            "",
            "## Test Metrics",
            f"- balanced_accuracy: {selected_test['balanced_accuracy']:.3f}",
            f"- true precision/recall/f1: {selected_test['true_class']['precision']:.3f} / {selected_test['true_class']['recall']:.3f} / {selected_test['true_class']['f1']:.3f}",
            f"- false precision/recall/f1: {selected_test['negative_class']['precision']:.3f} / {selected_test['negative_class']['recall']:.3f} / {selected_test['negative_class']['f1']:.3f}",
            f"- negative_recall: {selected_test['negative_class']['recall']:.3f}",
            f"- false_negative_rate: {selected_test['false_negative_rate']:.3f}",
            f"- macro_f1: {selected_test['macro_f1']:.3f}",
            f"- negative_pr_auc: {selected_test['negative_pr_auc']}",
            f"- roc_auc: {selected_test['roc_auc']}",
            f"- confusion_matrix labels [false, true]: {selected_test['confusion_matrix']['matrix']}",
            "",
            "## Per-Source Test Metrics",
            *(per_source_lines or ["- Not enough per-source samples to summarize."]),
            "",
            "## Artifacts",
            f"- model: {model_path}",
            "- feature_schema.json",
            "- metrics.json",
            "- training_report.md",
            "",
            "## Integration Note",
            "The saved model exposes UtilityValidatorModel.load(...).predict(...), plus an evaluate(...) method matching the Utility Preservation Gate protocol for replay experiments. It is not wired into the main flow by default.",
            "",
        ]
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} in {path} is not an object")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _paths_from_expanded_summary(repo: Path, dataset_root: Path) -> dict[str, dict[str, str]]:
    summary_path = dataset_root / "reports" / "expanded_build_summary.json"
    if not summary_path.exists():
        return {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    outputs = summary.get("outputs") if isinstance(summary, Mapping) else {}
    if not isinstance(outputs, Mapping):
        return {}
    result: dict[str, dict[str, str]] = {}
    for key in ("feature_aliases", "features", "labels"):
        value = outputs.get(key)
        if isinstance(value, Mapping):
            result[key] = {str(split): str(path) for split, path in value.items()}
    return result


def _first_existing(paths: Sequence[Path], description: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"Could not find {description}; tried {joined}")


def _resolve_under_repo(repo: Path, path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else repo / raw


def _assert_expanded_safe_path(path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    if "smoke" in lowered or "smoke_real" in lowered or "fake" in lowered:
        raise ValueError(f"Refusing to use smoke/fake data path: {path}")
    if not any(part.lower().startswith("expanded_") for part in path.parts):
        raise ValueError(f"Refusing non-expanded data path: {path}")


def _validate_prompt_safe_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    for row_index, row in enumerate(rows, start=1):
        if row.get("prompt_text_included") is True:
            raise ValueError(f"prompt_text_included=true row found in {path}:{row_index}")
        materialization = row.get("prompt_materialization")
        if isinstance(materialization, Mapping) and materialization.get("prompt_text_included") is True:
            raise ValueError(f"prompt_materialization.prompt_text_included=true row found in {path}:{row_index}")
        present_prompt_fields = [field for field in PROMPT_TEXT_FIELDS if field in row and row.get(field) not in (None, "")]
        if present_prompt_fields:
            joined = ", ".join(present_prompt_fields)
            raise ValueError(f"prompt text/body field(s) found in {path}:{row_index}: {joined}")


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return (value,)


def _as_string(value: Any) -> str:
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    return str(raw)


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        if isinstance(value, bool):
            return float(int(value))
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _scope_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    if raw == "agent":
        return "agent_local"
    if raw is None:
        return ""
    return str(raw)


def _placement_to_feature_dict(placement: Any) -> dict[str, Any]:
    return {
        "moved": bool(getattr(placement, "moved", False)),
        "original_scope": _scope_value(getattr(placement, "original_scope", None)),
        "target_scope": _scope_value(getattr(placement, "target_scope", None)),
        "risk_tags": list(getattr(placement, "risk_tags", ()) or ()),
        "dependency_notes": list(getattr(placement, "dependency_notes", ()) or ()),
        "cache_contribution": float(getattr(placement, "cache_contribution", 0.0) or 0.0),
        "placement_score": float(getattr(placement, "placement_score", 0.0) or 0.0),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Utility Validator baseline_v0 from expanded real labels.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--dataset-root", default="datasets/utility_validator")
    parser.add_argument("--output-dir", default="artifacts/utility_validator/baseline_v0")
    parser.add_argument("--logistic-iterations", type=int, default=3000)
    parser.add_argument("--random-forest-estimators", type=int, default=80)
    args = parser.parse_args(argv)
    result = train_utility_validator_baseline_v0(
        repo_root=args.repo_root,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        logistic_iterations=args.logistic_iterations,
        random_forest_estimators=args.random_forest_estimators,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "metrics"}, indent=2, ensure_ascii=False))
    return 0


def _register_pickle_module_alias() -> None:
    """Keep pickles loadable when this file is executed with python -m."""
    module_name = __spec__.name if __spec__ is not None and __spec__.name else None
    if __name__ != "__main__" or not module_name:
        return
    sys.modules[module_name] = sys.modules[__name__]
    for cls in (
        UtilityValidatorModel,
        StructuredFeatureEncoder,
        LogisticRegressionBaseline,
        DecisionTreeBaseline,
        RandomForestBaseline,
        _TreeNode,
        SplitPaths,
        LoadedSplit,
        LoadedDataset,
    ):
        cls.__module__ = module_name


if __name__ == "__main__":
    _register_pickle_module_alias()
    raise SystemExit(main())
