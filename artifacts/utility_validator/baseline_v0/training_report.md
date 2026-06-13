# Utility Validator baseline_v0 Training Report

baseline_v0 is a first structured-feature baseline. It is not the final Utility Validator; replay validation and main-flow Utility Preservation Gate integration still need follow-up verification.

## Data
- expanded_train: features=160, labels=240, skipped=80, distribution={'False': 40, 'True': 120}, feature_path=datasets\utility_validator\features\expanded_train\train_features.jsonl, label_path=datasets\utility_validator\labels\expanded_train\utility_labels.jsonl
- expanded_valid: features=20, labels=30, skipped=10, distribution={'False': 5, 'True': 15}, feature_path=datasets\utility_validator\features\expanded_valid\valid_features.jsonl, label_path=datasets\utility_validator\labels\expanded_valid\utility_labels.jsonl
- expanded_test: features=20, labels=31, skipped=11, distribution={'False': 5, 'True': 15}, feature_path=datasets\utility_validator\features\expanded_test\test_features.jsonl, label_path=datasets\utility_validator\labels\expanded_test\utility_labels.jsonl
- trainable samples: 200
- skipped labels excluded from training: 101
- overall label distribution: {'False': 50, 'None': 101, 'True': 150}

## Input Features
- categorical: dataset_source, scenario_type, candidate_strategy
- multi-value: source_scopes, target_scopes, risk_tags, dependency_notes
- numeric/static placement features: moved_block_count, global_prefix_tokens_or_chars, subgroup_prefix_tokens_or_chars, estimated_cache_gain, hard_warning_count, movement_distance_summary.max_abs_distance, movement_distance_summary.moved_position_count, movement_distance_summary.sum_abs_distance, placement_changes.count, placement_changes.moved_count, placement_changes.global_target_count, placement_changes.subgroup_target_count, placement_changes.agent_local_target_count, placement_changes.cache_contribution_sum, placement_changes.placement_score_avg, placement_changes.placement_score_max, placement_changes.dependency_note_count, placement_changes.risk_tag_count, cache_gain_report.longest_common_prefix_tokens, cache_gain_report.global_prefix_tokens, cache_gain_report.subgroup_prefix_tokens, cache_gain_report.estimated_cache_gain, cache_gain_report.node_count, cache_gain_report.node_token_len_sum, cache_gain_report.node_cache_contribution_sum, cache_gain_report.placement_cache_contribution_sum
- prompt text used: false

## Excluded Leakage Fields
is_utility_preserved, failure_type, original_run_status, reordered_run_status, label_source, api_model, input_tokens, output_tokens, cached_tokens, cached_tokens_delta, latency, latency_seconds, estimated_cost, estimated_cost_usd, whether_label_is_utility_verified, api_cost_estimate, api_run_reports, oracle_reports, validator_report, utility_status

## Models
- Logistic Regression with class_weight="balanced" implemented as weighted logistic loss.
- Random Forest with balanced bootstrap samples.
- final selected model by validation score: random_forest_balanced

## Validation Metrics
- balanced_accuracy: 0.900
- true precision/recall/f1: 1.000 / 0.800 / 0.889
- false precision/recall/f1: 0.625 / 1.000 / 0.769
- negative_recall: 1.000
- false_negative_rate: 0.000
- macro_f1: 0.829
- negative_pr_auc: 0.7367857142857144
- roc_auc: 0.8933333333333333
- confusion_matrix labels [false, true]: [[5, 0], [3, 12]]

## Test Metrics
- balanced_accuracy: 0.800
- true precision/recall/f1: 0.923 / 0.800 / 0.857
- false precision/recall/f1: 0.571 / 0.800 / 0.667
- negative_recall: 0.800
- false_negative_rate: 0.200
- macro_f1: 0.762
- negative_pr_auc: 0.8015873015873016
- roc_auc: 0.8866666666666667
- confusion_matrix labels [false, true]: [[4, 1], [3, 12]]

## Per-Source Test Metrics
- format_protocol: n=6, balanced_accuracy=0.500, negative_recall=1.000, false_negative_rate=0.000
- history_state: n=2, balanced_accuracy=0.500, negative_recall=0.000, false_negative_rate=0.000
- humaneval: n=2, balanced_accuracy=1.000, negative_recall=1.000, false_negative_rate=0.000
- role_privacy: n=3, balanced_accuracy=0.500, negative_recall=0.000, false_negative_rate=0.000
- synthetic_agent: n=7, balanced_accuracy=0.500, negative_recall=0.000, false_negative_rate=1.000

## Artifacts
- model: artifacts\utility_validator\baseline_v0\model.pkl
- feature_schema.json
- metrics.json
- training_report.md

## Integration Note
The saved model exposes UtilityValidatorModel.load(...).predict(...), plus an evaluate(...) method matching the Utility Preservation Gate protocol for replay experiments. It is not wired into the main flow by default.
