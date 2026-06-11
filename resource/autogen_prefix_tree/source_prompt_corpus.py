from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .dataset_eval import evaluate_dataset
from .ir import stable_hash
from .semantic_candidates import extract_semantic_candidates


@dataclass(frozen=True)
class SourcePromptRecord:
    source_path: str
    symbol: str
    extraction: str
    content: str


@dataclass(frozen=True)
class SourcePromptCorpusResult:
    source_root: str
    request_path: str
    summary_path: str
    telemetry_path: str
    prompt_count: int
    source_file_count: int
    extraction_counts: dict[str, int]
    summary: dict[str, Any]


def evaluate_source_prompt_corpus(
    *,
    source_root: str | Path,
    output_requests_path: str | Path,
    summary_path: str | Path,
    telemetry_path: str | Path,
    candidates_path: str | Path | None = None,
    session_id: str = "source-prompt-corpus",
    include_tests: bool = False,
    include_candidate_text: bool = False,
    min_prompt_chars: int = 40,
    enable_natural_language_segmentation: bool = False,
) -> SourcePromptCorpusResult:
    root = Path(source_root)
    records = tuple(
        collect_source_prompts(root, include_tests=include_tests, min_prompt_chars=min_prompt_chars)
    )
    request_target = Path(output_requests_path)
    _write_requests_jsonl(request_target, records, session_id=session_id)
    if candidates_path is not None:
        _write_candidate_jsonl(
            Path(candidates_path),
            records,
            include_candidate_text=include_candidate_text,
        )
    evaluation = evaluate_dataset(
        input_path=request_target,
        telemetry_path=telemetry_path,
        summary_path=summary_path,
        session_id=session_id,
        enable_natural_language_segmentation=enable_natural_language_segmentation,
    )
    source_files = {record.source_path for record in records}
    summary = dict(evaluation.summary)
    summary["source_prompt_corpus"] = _build_corpus_metadata(records, root)
    _write_json(Path(summary_path), summary)
    return SourcePromptCorpusResult(
        source_root=str(root),
        request_path=str(request_target),
        summary_path=str(summary_path),
        telemetry_path=str(telemetry_path),
        prompt_count=len(records),
        source_file_count=len(source_files),
        extraction_counts=dict(sorted(Counter(record.extraction for record in records).items())),
        summary=summary,
    )


def collect_source_prompts(
    source_root: str | Path,
    *,
    include_tests: bool = False,
    min_prompt_chars: int = 40,
) -> Iterable[SourcePromptRecord]:
    root = Path(source_root)
    seen: set[tuple[str, str, str]] = set()
    for path in sorted(root.rglob("*.py")):
        if not include_tests and _is_test_path(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        collector = _PromptCollector(path=path, min_prompt_chars=min_prompt_chars)
        collector.visit(tree)
        for record in collector.records:
            key = (record.source_path, record.symbol, record.content)
            if key in seen:
                continue
            seen.add(key)
            yield record
    for path in sorted(root.rglob("*.json")):
        if not include_tests and _is_test_path(path):
            continue
        for record in _collect_json_prompts(path=path, min_prompt_chars=min_prompt_chars):
            key = (record.source_path, record.symbol, record.content)
            if key in seen:
                continue
            seen.add(key)
            yield record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a prompt-safe coverage report from prompt-like strings in a local framework source tree."
    )
    parser.add_argument("--source-root", required=True, help="Local source tree to scan.")
    parser.add_argument("--requests", required=True, help="Output OpenAI-compatible request JSONL.")
    parser.add_argument("--summary", required=True, help="Output dataset_eval summary JSON.")
    parser.add_argument("--telemetry", required=True, help="Output prompt-safe telemetry JSONL.")
    parser.add_argument("--candidates", help="Optional semantic candidate JSONL for local-model or human labeling.")
    parser.add_argument(
        "--include-candidate-text",
        action="store_true",
        help="Include raw candidate text in --candidates output. Default stores hashes only.",
    )
    parser.add_argument("--session-id", default="source-prompt-corpus")
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument("--min-prompt-chars", type=int, default=40)
    parser.add_argument(
        "--enable-natural-language-segmentation",
        action="store_true",
        help="Opt-in offline analysis mode for unmarked natural-language system prompts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_source_prompt_corpus(
        source_root=args.source_root,
        output_requests_path=args.requests,
        summary_path=args.summary,
        telemetry_path=args.telemetry,
        candidates_path=args.candidates,
        session_id=args.session_id,
        include_tests=args.include_tests,
        include_candidate_text=args.include_candidate_text,
        min_prompt_chars=args.min_prompt_chars,
        enable_natural_language_segmentation=args.enable_natural_language_segmentation,
    )
    print(
        json.dumps(
            {
                "source_root": result.source_root,
                "request_path": result.request_path,
                "summary_path": result.summary_path,
                "telemetry_path": result.telemetry_path,
                "prompt_count": result.prompt_count,
                "source_file_count": result.source_file_count,
                "extraction_counts": result.extraction_counts,
                "summary": result.summary,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


class _PromptCollector(ast.NodeVisitor):
    def __init__(self, *, path: Path, min_prompt_chars: int) -> None:
        self.path = path
        self.min_prompt_chars = min_prompt_chars
        self.scope: list[str] = []
        self.function_depth = 0
        self.records: list[SourcePromptRecord] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.scope.append(node.name)
        self.function_depth += 1
        self._visit_argument_defaults(node.args)
        self.generic_visit(node)
        self.function_depth -= 1
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.scope.append(node.name)
        self.function_depth += 1
        self._visit_argument_defaults(node.args)
        self.generic_visit(node)
        self.function_depth -= 1
        self.scope.pop()

    def visit_Assign(self, node: ast.Assign) -> Any:
        value = _literal_string(node.value)
        if value is not None:
            for target in node.targets:
                name = _target_name(target)
                if name and _looks_prompt_name(name):
                    self._add(name, "prompt_named_constant", value)
                elif name and self.function_depth == 0:
                    self._add(name, "prompt_text_constant", value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        value = _literal_string(node.value)
        name = _target_name(node.target)
        if value is not None and name and _looks_prompt_name(name):
            self._add(name, "prompt_named_constant", value)
        elif value is not None and name and self.function_depth == 0:
            self._add(name, "prompt_text_constant", value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        for keyword in node.keywords:
            if keyword.arg and _looks_prompt_keyword(keyword.arg):
                value = _literal_string(keyword.value)
                if value is not None:
                    self._add(keyword.arg, f"keyword:{keyword.arg}", value)
        self.generic_visit(node)

    def _visit_argument_defaults(self, args: ast.arguments) -> None:
        positional = [*args.posonlyargs, *args.args]
        default_offset = len(positional) - len(args.defaults)
        for index, default in enumerate(args.defaults):
            arg = positional[default_offset + index]
            self._add_argument_default(arg.arg, default)
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is not None:
                self._add_argument_default(arg.arg, default)

    def _add_argument_default(self, name: str, node: ast.AST) -> None:
        if not _looks_prompt_name(name):
            return
        value = _literal_string(node)
        if value is not None:
            self._add(name, f"argument_default:{name}", value)

    def _add(self, symbol: str, extraction: str, content: str) -> None:
        stripped = content.strip()
        if not _looks_like_prompt_text(
            stripped,
            min_prompt_chars=self.min_prompt_chars,
            allow_short=_looks_strong_prompt_name(symbol),
        ):
            return
        qualified = ".".join((*self.scope, symbol)) if self.scope else symbol
        self.records.append(
            SourcePromptRecord(
                source_path=str(self.path),
                symbol=qualified,
                extraction=extraction,
                content=stripped,
            )
        )


def _write_requests_jsonl(path: Path, records: Sequence[SourcePromptRecord], *, session_id: str) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            body = {
                "messages": [
                    {"role": "system", "content": record.content},
                    {"role": "user", "content": "Evaluate this prompt structure for offline prefix coverage."},
                ],
                "source_prompt_metadata": {
                    "source_path": record.source_path,
                    "symbol": record.symbol,
                    "extraction": record.extraction,
                    "content_hash": stable_hash({"text": record.content}),
                },
            }
            row = {
                "id": f"{session_id}:{index:06d}",
                "session_id": session_id,
                "body": body,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_candidate_jsonl(
    path: Path,
    records: Sequence[SourcePromptRecord],
    *,
    include_candidate_text: bool,
) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            parent_hash = stable_hash({"text": record.content})
            for candidate in extract_semantic_candidates(record.content, parent_hash=parent_hash):
                row: dict[str, Any] = {
                    "schema_version": "prefix-semantic-candidate-v1",
                    "source_path": record.source_path,
                    "symbol": record.symbol,
                    "extraction": record.extraction,
                    "parent_hash": candidate.parent_hash,
                    "candidate_id": candidate.candidate_id,
                    "semantic_hint": candidate.semantic_hint,
                    "confidence": candidate.confidence,
                    "char_count": candidate.char_count,
                    "line_count": candidate.line_count,
                    "text_hash": candidate.text_hash,
                    "risk_tags": candidate.risk_tags,
                    "label": None,
                    "label_source": None,
                    "label_notes": None,
                }
                if include_candidate_text:
                    row["text"] = _candidate_text(record.content, candidate.text_hash)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _candidate_text(prompt: str, text_hash: str) -> str:
    for segment in _candidate_segments(prompt):
        if stable_hash({"semantic_candidate": segment}) == text_hash:
            return segment
    return ""


def _candidate_segments(prompt: str) -> Iterable[str]:
    paragraphs = [paragraph.strip() for paragraph in prompt.split("\n\n") if paragraph.strip()]
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(paragraph) <= 220 or len(lines) <= 1:
            yield paragraph
            continue
        for line in lines:
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if cleaned:
                yield cleaned


def _build_corpus_metadata(records: Sequence[SourcePromptRecord], root: Path) -> dict[str, Any]:
    sources = []
    for record in records:
        try:
            relative_path = str(Path(record.source_path).relative_to(root))
        except ValueError:
            relative_path = record.source_path
        sources.append(
            {
                "source_path": relative_path,
                "symbol": record.symbol,
                "extraction": record.extraction,
                "content_hash": stable_hash({"text": record.content}),
                "char_count": len(record.content),
                "line_count": len(record.content.splitlines()),
            }
        )
    return {
        "prompt_count": len(records),
        "source_file_count": len({record.source_path for record in records}),
        "extraction_counts": dict(sorted(Counter(record.extraction for record in records).items())),
        "prompt_sources": sources,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _literal_string(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_string(node.left)
        right = _literal_string(node.right)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                parts.append("{" + _format_fstring_expression(value) + "}")
            else:
                return None
        return "".join(parts)
    return None


def _format_fstring_expression(value: ast.FormattedValue) -> str:
    try:
        expression = ast.unparse(value.value)
    except Exception:  # pragma: no cover - ast.unparse is best-effort here.
        expression = "expr"
    if value.format_spec is None:
        return expression
    spec = _literal_string(value.format_spec)
    return f"{expression}:{spec}" if spec else expression


def _target_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _looks_prompt_name(name: str) -> bool:
    lowered = name.lower()
    return any(
        marker in lowered
        for marker in (
            "system_message",
            "system_prompt",
            "selector_prompt",
            "sys_prompt",
            "prompt",
            "instructions",
            "instruction",
        )
    )


def _looks_prompt_keyword(name: str) -> bool:
    lowered = name.lower()
    if _looks_prompt_name(lowered):
        return True
    return lowered in {"text", "content"}


def _looks_strong_prompt_name(name: str) -> bool:
    lowered = name.lower()
    return any(
        marker in lowered
        for marker in (
            "system_message",
            "system_prompt",
            "selector_prompt",
            "sys_prompt",
            "instructions",
            "instruction",
            "conversation_history_prompt",
        )
    )


def _looks_like_prompt_text(text: str, *, min_prompt_chars: int, allow_short: bool = False) -> bool:
    if not text:
        return False
    lowered = text.lower()
    short_markers = (
        "you are",
        "assistant",
        "system prompt",
        "<system-info",
        "<system-hint",
        "current task",
        "final answer",
        "thought:",
        "action:",
        "observation:",
    )
    if allow_short and len(text) >= 12 and any(marker in lowered for marker in short_markers):
        return True
    if len(text) < min_prompt_chars:
        return False
    return any(
        marker in lowered
        for marker in (
            *short_markers,
            "{history}",
            "{roles}",
            "<history>",
            "conversation history",
            "expected criteria",
            "response format",
            "respond using",
            "you must",
            "you need",
            "your options",
            "you only have access",
            "available tools",
            "tool_names",
            "agent skill",
            "must read",
            "terminate",
        )
    )


def _collect_json_prompts(*, path: Path, min_prompt_chars: int) -> Iterable[SourcePromptRecord]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    def walk(value: Any, key_path: tuple[str, ...]) -> Iterable[SourcePromptRecord]:
        if isinstance(value, str):
            stripped = value.strip()
            symbol = ".".join(key_path) if key_path else "$"
            allow_short = _looks_strong_prompt_name(symbol)
            if not _looks_json_prompt_path(key_path):
                return
            if _looks_like_prompt_text(
                stripped,
                min_prompt_chars=min_prompt_chars,
                allow_short=allow_short,
            ):
                yield SourcePromptRecord(
                    source_path=str(path),
                    symbol=symbol,
                    extraction="json_prompt_value",
                    content=stripped,
                )
            return
        if isinstance(value, dict):
            for key, child in value.items():
                yield from walk(child, (*key_path, str(key)))
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(child, (*key_path, str(index)))

    yield from walk(data, ())


def _looks_json_prompt_path(key_path: tuple[str, ...]) -> bool:
    lowered = tuple(part.lower() for part in key_path)
    # JSON config files often mix prompts with UI schema descriptions and tool source.
    # Keep explicit prompt fields and top-level prompt resource namespaces only.
    if lowered and lowered[-1] in {"source_code", "code", "description"}:
        return False
    if any(
        part
        in {
            "system_message",
            "system_prompt",
            "sys_prompt",
            "selector_prompt",
            "prompt",
            "instructions",
            "instruction",
            "backstory",
            "goal",
        }
        for part in lowered
    ):
        return True
    if not lowered:
        return False
    return lowered[0] in {
        "slices",
        "errors",
        "tools",
        "memory",
        "reasoning",
        "planning",
    }


def _is_test_path(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    return "tests" in lowered_parts or "test" in lowered_parts


if __name__ == "__main__":
    raise SystemExit(main())
