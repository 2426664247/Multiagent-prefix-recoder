from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

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


@dataclass(frozen=True)
class OpenAICompatibleSemanticGuardConfig:
    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 30.0
    min_confidence: float = 0.75
    max_message_chars: int = 12000


class OpenAICompatibleSemanticGuard:
    """Local OpenAI-compatible semantic invariant judge.

    The guard is intended for local/sandboxed model services. It may send prompt
    text to the configured local endpoint, but its returned report is prompt-safe
    and contains only aggregate judgment metadata.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        min_confidence: float = 0.75,
        max_message_chars: int = 12000,
    ) -> None:
        self.config = OpenAICompatibleSemanticGuardConfig(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            min_confidence=min_confidence,
            max_message_chars=max_message_chars,
        )

    def evaluate(
        self,
        *,
        original_messages: Sequence[LLMMessage],
        rewritten_messages: Sequence[LLMMessage],
        compile_result: CompileResult,
        plan: PrefixPlan,
    ) -> SemanticGuardReport:
        try:
            raw_result = self._call_model(
                original_messages=original_messages,
                rewritten_messages=rewritten_messages,
                compile_result=compile_result,
                plan=plan,
            )
            passed, reason, checks, confidence = self._parse_result(raw_result)
        except Exception as exc:  # noqa: BLE001
            return SemanticGuardReport(
                passed=False,
                reason=f"local_judge_error:{type(exc).__name__}",
                checks=("local_semantic_judge",),
                model_name=self.config.model,
                confidence=None,
            )

        if passed and confidence is not None and confidence < self.config.min_confidence:
            return SemanticGuardReport(
                passed=False,
                reason="local_judge_low_confidence",
                checks=checks,
                model_name=self.config.model,
                confidence=confidence,
            )
        return SemanticGuardReport(
            passed=passed,
            reason=reason,
            checks=checks,
            model_name=self.config.model,
            confidence=confidence,
        )

    def _call_model(
        self,
        *,
        original_messages: Sequence[LLMMessage],
        rewritten_messages: Sequence[LLMMessage],
        compile_result: CompileResult,
        plan: PrefixPlan,
    ) -> Mapping[str, Any]:
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a semantic safety judge for a prefix-reordering cache plugin. "
                        "The plugin may move stable shared prompt blocks earlier to improve provider cache reuse. "
                        "Return JSON only with keys passed, reason, checks, and confidence. "
                        "passed must be true only when agent identity, latest user instruction, private/public boundary, "
                        "tool-call/tool-result binding, output format, and temporal causality are preserved. "
                        "If uncertain, return passed=false."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        self._judge_payload(
                            original_messages=original_messages,
                            rewritten_messages=rewritten_messages,
                            compile_result=compile_result,
                            plan=plan,
                        ),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(
            _chat_completions_url(self.config.base_url),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "\n".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
        if not isinstance(content, str):
            raise ValueError("missing_model_content")
        parsed = _parse_json_object(content)
        if not isinstance(parsed, Mapping):
            raise ValueError("model_response_not_object")
        return parsed

    def _judge_payload(
        self,
        *,
        original_messages: Sequence[LLMMessage],
        rewritten_messages: Sequence[LLMMessage],
        compile_result: CompileResult,
        plan: PrefixPlan,
    ) -> dict[str, Any]:
        blocks_by_id = {block.block_id: block for block in compile_result.blocks}
        return {
            "message_types_before": [_message_type(message) for message in original_messages],
            "message_types_after": [_message_type(message) for message in rewritten_messages],
            "moved_blocks": [
                {
                    "semantic_type": blocks_by_id[block_id].semantic_type.value,
                    "movability": blocks_by_id[block_id].movability.value,
                    "share_scope": blocks_by_id[block_id].share_scope.value,
                    "source_role": blocks_by_id[block_id].source_role,
                    "content_hash": blocks_by_id[block_id].content_hash,
                    "char_count": len(blocks_by_id[block_id].rendered_text or ""),
                }
                for block_id in plan.moved_blocks
                if block_id in blocks_by_id
            ],
            "cacheable_prefix_semantic_types": [
                blocks_by_id[block_id].semantic_type.value
                for block_id in plan.cacheable_prefix_blocks
                if block_id in blocks_by_id
            ],
            "original_messages": [self._message_payload(message) for message in original_messages],
            "rewritten_messages": [self._message_payload(message) for message in rewritten_messages],
        }

    def _message_payload(self, message: LLMMessage) -> dict[str, Any]:
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        return {
            "type": _message_type(message),
            "source": getattr(message, "source", None),
            "content": _truncate(content, self.config.max_message_chars),
            "truncated": len(content) > self.config.max_message_chars,
        }

    def _parse_result(self, value: Mapping[str, Any]) -> tuple[bool, str, tuple[str, ...], float | None]:
        passed = _parse_passed(value)
        reason = _safe_reason(value.get("reason") or ("passed" if passed else "rejected"))
        checks = tuple(_safe_reason(check) for check in value.get("checks") or ("local_semantic_judge",))
        confidence = _float_or_none(value.get("confidence"))
        return passed, reason, checks, confidence


def _parse_passed(value: Mapping[str, Any]) -> bool:
    raw_passed = value.get("passed")
    if isinstance(raw_passed, bool):
        return raw_passed
    if isinstance(raw_passed, str):
        normalized = raw_passed.strip().lower()
        if normalized in {"true", "pass", "passed", "accept", "accepted", "yes"}:
            return True
        if normalized in {"false", "fail", "failed", "reject", "rejected", "review", "no"}:
            return False
    label = str(value.get("label") or "").strip().lower()
    if label == "accept":
        return True
    if label in {"review", "reject"}:
        return False
    raise ValueError("missing_passed")


def _message_type(message: LLMMessage) -> str:
    return getattr(message, "type", type(message).__name__)


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model_response_not_object")
    return value


def _safe_reason(value: Any) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(value).strip().lower()).strip("_")
    return cleaned[:80] or "local_semantic_judge"


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _truncate(value: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars]
