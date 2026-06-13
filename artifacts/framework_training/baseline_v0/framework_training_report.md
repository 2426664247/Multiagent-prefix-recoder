# Framework Training baseline_v0 Report

baseline_v0 is a replay and shadow validation baseline, not a final enabled policy.

## Validator Shadow
- samples: 20
- shadow decisions: {'observe': 2, 'would_accept': 12, 'would_reject': 6}
- model gate if enabled: {'accepted_count': 12, 'rejected_count': 6, 'observe_count': 2, 'false_kill_risk_count': 3, 'leak_risk_count': 1}

## Planner Ranker
- samples: 20
- top_5: {'priority_count': 5, 'unsafe_count': 0}

## Replay Checks
- shadow preserves default decision: True
- risk summary: {'false_kill_risk_count': 3, 'leak_risk_count': 1, 'planner_top_5_unsafe_count': 0}

## Artifacts
- artifacts\framework_training\baseline_v0\validator_shadow_report.json
- artifacts\framework_training\baseline_v0\planner_ranker_report.json
- artifacts\framework_training\baseline_v0\framework_replay_report.json
- artifacts\framework_training\baseline_v0\framework_training_report.md
