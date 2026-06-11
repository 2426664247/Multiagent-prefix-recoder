from .client import PrefixReorderClient
from .compiler import LocalPromptCompiler
from .ir import (
    BlockPosition,
    CompileResult,
    Movability,
    PrefixTree,
    PrefixTreeNode,
    PromptBlock,
    SemanticType,
    ShareScope,
)
from .planner import HierarchicalPrefixPlanner, PrefixPlan, rewrite_messages
from .request_capture import JsonlRequestLogger, RequestCaptureClient
from .semantic_guard import OpenAICompatibleSemanticGuard, OpenAICompatibleSemanticGuardConfig, SemanticGuard, SemanticGuardReport
from .static_client import StaticResponseClient
from .telemetry import JsonlTelemetryLogger, TelemetrySink, TelemetrySummary, load_jsonl_telemetry, summarize_telemetry
from .validator import CacheUtilityEstimate, CacheUtilityValidator, ValidationReport

__all__ = [
    "BlockPosition",
    "CacheUtilityEstimate",
    "CacheUtilityValidator",
    "CompileResult",
    "HierarchicalPrefixPlanner",
    "JsonlRequestLogger",
    "JsonlTelemetryLogger",
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
    "PrefixPlan",
    "PrefixReorderClient",
    "RequestCaptureClient",
    "PrefixTree",
    "PrefixTreeNode",
    "PromptBlock",
    "SemanticType",
    "SemanticGuard",
    "SemanticGuardReport",
    "ShareScope",
    "StaticResponseClient",
    "TelemetrySink",
    "TelemetrySummary",
    "ValidationReport",
    "load_jsonl_telemetry",
    "build_local_judge_goldset_template",
    "export_goldset_template_csv",
    "import_goldset_labeled_csv",
    "run_local_judge_healthcheck",
    "run_local_judge_quality_eval",
    "rewrite_messages",
    "rewrite_openai_request_body",
    "summarize_telemetry",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
