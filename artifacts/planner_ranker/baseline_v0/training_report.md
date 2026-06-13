# Planner Ranker baseline_v0 Training Report

baseline_v0 is a first structured-feature planner/ranking baseline. It is not a final planner policy and is intended for replay and controlled shadow validation.

## Data
- expanded_train: samples=160, target_distribution={'False': 48, 'True': 112}, utility_distribution={'False': 40, 'True': 120}
- expanded_valid: samples=20, target_distribution={'False': 5, 'True': 15}, utility_distribution={'False': 5, 'True': 15}
- expanded_test: samples=20, target_distribution={'False': 5, 'True': 15}, utility_distribution={'False': 5, 'True': 15}
- trainable samples: 200
- cache gain priority threshold: 243.0

## Target
- positive iff is_utility_preserved is true and estimated_cache_gain is at least the expanded_train safe positive median
- utility labels are used only to build the target, never as inference inputs
- prompt text used: false

## Input Features
- categorical: dataset_source, scenario_type, candidate_strategy
- multi-value: source_scopes, target_scopes, risk_tags, dependency_notes
- numeric/static placement features: moved_block_count, global_prefix_tokens_or_chars, subgroup_prefix_tokens_or_chars, estimated_cache_gain, hard_warning_count, movement_distance_summary.max_abs_distance, movement_distance_summary.moved_position_count, movement_distance_summary.sum_abs_distance, placement_changes.count, placement_changes.moved_count, placement_changes.global_target_count, placement_changes.subgroup_target_count, placement_changes.agent_local_target_count, placement_changes.cache_contribution_sum, placement_changes.placement_score_avg, placement_changes.placement_score_max, placement_changes.dependency_note_count, placement_changes.risk_tag_count, cache_gain_report.longest_common_prefix_tokens, cache_gain_report.global_prefix_tokens, cache_gain_report.subgroup_prefix_tokens, cache_gain_report.estimated_cache_gain, cache_gain_report.node_count, cache_gain_report.node_token_len_sum, cache_gain_report.node_cache_contribution_sum, cache_gain_report.placement_cache_contribution_sum

## Excluded Leakage Fields
is_utility_preserved, failure_type, original_run_status, reordered_run_status, label_source, api_model, input_tokens, output_tokens, cached_tokens, cached_tokens_delta, latency, latency_seconds, estimated_cost, estimated_cost_usd, whether_label_is_utility_verified, api_cost_estimate, api_run_reports, oracle_reports, validator_report, utility_status

## Models
- Logistic Regression with balanced weighted logistic loss.
- Random Forest with balanced bootstrap samples.
- final selected model by validation score: random_forest_balanced

## Validation Metrics
- balanced_accuracy: 0.800
- priority precision/recall/f1: 0.882 / 1.000 / 0.938
- unsafe_rejection_rate: 0.600
- unsafe_prioritized_count: 2
- top_3_priority_rate: 1.000

## Test Metrics
- balanced_accuracy: 0.767
- priority precision/recall/f1: 0.875 / 0.933 / 0.903
- unsafe_rejection_rate: 0.600
- unsafe_prioritized_count: 2
- top_3_priority_rate: 1.000

## Artifacts
- model: artifacts\planner_ranker\baseline_v0\model.pkl
- feature_schema.json
- metrics.json
- training_report.md
