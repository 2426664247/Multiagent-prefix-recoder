from .client import PrefixReorderClient
from .cache_estimator import (
    CacheEstimateReport,
    CacheEstimator,
    CacheHitProxyEstimator,
    PrefixTreeEstimator,
    ProviderTelemetryEstimator,
    resolve_cache_estimator,
)
from .compiler import LocalPromptCompiler
from .feedback import (
    JsonlFeedbackLogger,
    PlannerFeedbackLearner,
    PlannerFeedbackRecord,
    load_jsonl_feedback,
    summarize_feedback_records,
)
from .ir import (
    BlockPlacement,
    BlockPosition,
    CompileResult,
    Movability,
    PrefixScope,
    PrefixTree,
    PrefixTreeCandidate,
    PrefixTreeNode,
    PromptBlock,
    SemanticType,
    ShareScope,
)
from .planner import HierarchicalPrefixPlanner, PrefixPlan, rewrite_messages
from .replay import (
    ReplayRunStore,
    build_replay_run_record,
    export_planner_training_samples,
    export_utility_labeling_samples,
    load_replay_run,
    replay_candidate,
    revalidate_candidate,
)
from .request_capture import JsonlRequestLogger, RequestCaptureClient
from .semantic_guard import OpenAICompatibleSemanticGuard, OpenAICompatibleSemanticGuardConfig, SemanticGuard, SemanticGuardReport
from .static_client import StaticResponseClient
from .telemetry import JsonlTelemetryLogger, TelemetrySink, TelemetrySummary, load_jsonl_telemetry, summarize_telemetry
from .utility_oracles import (
    JsonSchemaOracle,
    OracleResult,
    PrivacyLeakOracle,
    RoleBoundaryOracle,
    StateConsistencyOracle,
    ToolTraceOracle,
    UnitTestOracle,
)
from .validator import (
    CacheUtilityEstimate,
    CacheUtilityValidator,
    UtilityModelGateReport,
    UtilityPreservationReport,
    ValidationReport,
)

__all__ = [
    "BlockPlacement",
    "BlockPosition",
    "CacheUtilityEstimate",
    "CacheUtilityValidator",
    "CacheEstimateReport",
    "CacheEstimator",
    "CacheHitProxyEstimator",
    "CompileResult",
    "HierarchicalPrefixPlanner",
    "JsonlRequestLogger",
    "JsonlFeedbackLogger",
    "JsonlTelemetryLogger",
    "JsonSchemaOracle",
    "GoldsetCsvResult",
    "LocalPromptCompiler",
    "LocalJudgeHealthcheckResult",
    "LocalJudgeGoldsetChainVerificationResult",
    "LocalJudgeGoldsetTemplateResult",
    "LocalJudgeGoldsetValidationResult",
    "LocalJudgeQualityEvalResult",
    "Movability",
    "OpenAICompatibleRequestAdapter",
    "OpenAIRequestRewriteResult",
    "OpenAICompatibleSemanticGuard",
    "OpenAICompatibleSemanticGuardConfig",
    "OracleResult",
    "PrefixPlan",
    "PlannerRankerModel",
    "PrefixTreeEstimator",
    "PrefixScope",
    "PrefixReorderClient",
    "PrefixTreeCandidate",
    "PlannerFeedbackLearner",
    "PlannerFeedbackRecord",
    "RequestCaptureClient",
    "ReplayRunStore",
    "PrefixTree",
    "PrefixTreeNode",
    "PrivacyLeakOracle",
    "ProviderTelemetryEstimator",
    "PromptBlock",
    "RoleBoundaryOracle",
    "SemanticType",
    "SemanticGuard",
    "SemanticGuardReport",
    "ShareScope",
    "StaticResponseClient",
    "StateConsistencyOracle",
    "TelemetrySink",
    "TelemetrySummary",
    "ToolTraceOracle",
    "UnitTestOracle",
    "UtilityDatasetBuildResult",
    "UtilityGoldsetTemplateResult",
    "UtilityModelGateReport",
    "UtilityValidatorModel",
    "UtilityPreservationReport",
    "ValidationReport",
    "load_jsonl_feedback",
    "load_jsonl_telemetry",
    "build_humaneval_utility_goldset_template",
    "build_utility_validator_smoke_dataset",
    "discover_expanded_dataset_paths",
    "build_replay_run_record",
    "build_local_judge_goldset_template",
    "export_goldset_template_csv",
    "import_goldset_labeled_csv",
    "run_local_judge_healthcheck",
    "run_local_judge_quality_eval",
    "rewrite_messages",
    "load_replay_run",
    "load_expanded_utility_validator_dataset",
    "load_planner_ranker_dataset",
    "replay_candidate",
    "revalidate_candidate",
    "export_utility_labeling_samples",
    "export_planner_training_samples",
    "rewrite_openai_request_body",
    "resolve_cache_estimator",
    "summarize_feedback_records",
    "summarize_telemetry",
    "train_utility_validator_baseline_v0",
    "train_planner_ranker_baseline_v0",
    "validate_local_judge_goldset",
    "verify_local_judge_goldset_chain",
]


def __getattr__(name: str):
    if name in {
        "LocalJudgeHealthcheckResult",
        "run_local_judge_healthcheck",
    }:
        from .local_judge_healthcheck import LocalJudgeHealthcheckResult, run_local_judge_healthcheck

        values = {
            "LocalJudgeHealthcheckResult": LocalJudgeHealthcheckResult,
            "run_local_judge_healthcheck": run_local_judge_healthcheck,
        }
        return values[name]
    if name in {
        "LocalJudgeGoldsetChainVerificationResult",
        "verify_local_judge_goldset_chain",
    }:
        from .local_judge_goldset_chain_verify import (
            LocalJudgeGoldsetChainVerificationResult,
            verify_local_judge_goldset_chain,
        )

        values = {
            "LocalJudgeGoldsetChainVerificationResult": LocalJudgeGoldsetChainVerificationResult,
            "verify_local_judge_goldset_chain": verify_local_judge_goldset_chain,
        }
        return values[name]
    if name in {
        "LocalJudgeGoldsetValidationResult",
        "validate_local_judge_goldset",
    }:
        from .local_judge_goldset_validate import LocalJudgeGoldsetValidationResult, validate_local_judge_goldset

        values = {
            "LocalJudgeGoldsetValidationResult": LocalJudgeGoldsetValidationResult,
            "validate_local_judge_goldset": validate_local_judge_goldset,
        }
        return values[name]
    if name in {
        "LocalJudgeQualityEvalResult",
        "run_local_judge_quality_eval",
    }:
        from .local_judge_quality_eval import LocalJudgeQualityEvalResult, run_local_judge_quality_eval

        values = {
            "LocalJudgeQualityEvalResult": LocalJudgeQualityEvalResult,
            "run_local_judge_quality_eval": run_local_judge_quality_eval,
        }
        return values[name]
    if name in {
        "OpenAICompatibleRequestAdapter",
        "OpenAIRequestRewriteResult",
        "rewrite_openai_request_body",
    }:
        from .openai_request_adapter import (
            OpenAICompatibleRequestAdapter,
            OpenAIRequestRewriteResult,
            rewrite_openai_request_body,
        )

        values = {
            "OpenAICompatibleRequestAdapter": OpenAICompatibleRequestAdapter,
            "OpenAIRequestRewriteResult": OpenAIRequestRewriteResult,
            "rewrite_openai_request_body": rewrite_openai_request_body,
        }
        return values[name]
    if name in {
        "LocalJudgeGoldsetTemplateResult",
        "build_local_judge_goldset_template",
    }:
        from .local_judge_goldset_builder import (
            LocalJudgeGoldsetTemplateResult,
            build_local_judge_goldset_template,
        )

        values = {
            "LocalJudgeGoldsetTemplateResult": LocalJudgeGoldsetTemplateResult,
            "build_local_judge_goldset_template": build_local_judge_goldset_template,
        }
        return values[name]
    if name in {
        "GoldsetCsvResult",
        "export_goldset_template_csv",
        "import_goldset_labeled_csv",
    }:
        from .local_judge_goldset_csv import (
            GoldsetCsvResult,
            export_goldset_template_csv,
            import_goldset_labeled_csv,
        )

        values = {
            "GoldsetCsvResult": GoldsetCsvResult,
            "export_goldset_template_csv": export_goldset_template_csv,
            "import_goldset_labeled_csv": import_goldset_labeled_csv,
        }
        return values[name]
    if name in {
        "UtilityGoldsetTemplateResult",
        "build_humaneval_utility_goldset_template",
    }:
        from .utility_goldset_builder import UtilityGoldsetTemplateResult, build_humaneval_utility_goldset_template

        values = {
            "UtilityGoldsetTemplateResult": UtilityGoldsetTemplateResult,
            "build_humaneval_utility_goldset_template": build_humaneval_utility_goldset_template,
        }
        return values[name]
    if name in {
        "UtilityDatasetBuildResult",
        "build_utility_validator_smoke_dataset",
    }:
        from .utility_dataset_builder import UtilityDatasetBuildResult, build_utility_validator_smoke_dataset

        values = {
            "UtilityDatasetBuildResult": UtilityDatasetBuildResult,
            "build_utility_validator_smoke_dataset": build_utility_validator_smoke_dataset,
        }
        return values[name]
    if name in {
        "UtilityValidatorModel",
        "discover_expanded_dataset_paths",
        "load_expanded_utility_validator_dataset",
        "train_utility_validator_baseline_v0",
    }:
        from .utility_validator_baseline import (
            UtilityValidatorModel,
            discover_expanded_dataset_paths,
            load_expanded_utility_validator_dataset,
            train_utility_validator_baseline_v0,
        )

        values = {
            "UtilityValidatorModel": UtilityValidatorModel,
            "discover_expanded_dataset_paths": discover_expanded_dataset_paths,
            "load_expanded_utility_validator_dataset": load_expanded_utility_validator_dataset,
            "train_utility_validator_baseline_v0": train_utility_validator_baseline_v0,
        }
        return values[name]
    if name in {
        "PlannerRankerModel",
        "load_planner_ranker_dataset",
        "train_planner_ranker_baseline_v0",
    }:
        from .planner_ranker_baseline import (
            PlannerRankerModel,
            load_planner_ranker_dataset,
            train_planner_ranker_baseline_v0,
        )

        values = {
            "PlannerRankerModel": PlannerRankerModel,
            "load_planner_ranker_dataset": load_planner_ranker_dataset,
            "train_planner_ranker_baseline_v0": train_planner_ranker_baseline_v0,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
