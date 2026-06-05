from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from autogen_core.models import LLMMessage

from .ir import CompileResult
from .planner import PrefixPlan


@dataclass(frozen=True)
class SemanticGuardReport:
    passed: bool
    reason: str
    checks: tuple[str, ...] = ()
    model_name: str | None = None
    confidence: float | None = None


class SemanticGuard(Protocol):
    def evaluate(
        self,
        *,
        original_messages: Sequence[LLMMessage],
        rewritten_messages: Sequence[LLMMessage],
        compile_result: CompileResult,
        plan: PrefixPlan,
    ) -> SemanticGuardReport:
        """返回语义 guard 判断；无法确定时应返回 passed=False。"""
