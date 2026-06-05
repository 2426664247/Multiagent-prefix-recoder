from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, Mapping, Optional, Sequence, Union

from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,  # type: ignore
    ModelFamily,
    ModelInfo,
    RequestUsage,
    SystemMessage,
)
from autogen_core.tools import Tool, ToolSchema
from pydantic import BaseModel

from .client import PrefixReorderClient
from .telemetry import load_jsonl_telemetry, summarize_telemetry


DEFAULT_AGENTS = ("planner", "engineer", "reviewer")


@dataclass(frozen=True)
class MicrobenchmarkResult:
    telemetry_path: str
    summary: dict[str, Any]


async def run_microbenchmark(
    *,
    telemetry_path: str | Path,
    summary_path: str | Path | None = None,
    agents: Sequence[str] = DEFAULT_AGENTS,
    repeats: int = 1,
    session_id: str = "microbench",
) -> MicrobenchmarkResult:
    telemetry_target = Path(telemetry_path)
    if telemetry_target.exists():
        telemetry_target.unlink()
    if summary_path is not None:
        summary_target = Path(summary_path)
        if summary_target.exists():
            summary_target.unlink()

    client = PrefixReorderClient(
        _NoopClient(),
        session_id=session_id,
        telemetry_log_path=telemetry_target,
    )
    for repeat_index in range(repeats):
        for agent in agents:
            await client.create(_messages(agent, repeat_index=repeat_index))

    records = load_jsonl_telemetry(telemetry_target)
    summary = summarize_telemetry(records).to_dict()
    summary["agent_count"] = len(tuple(agents))
    summary["repeats"] = repeats
    summary["session_id"] = session_id
    if summary_path is not None:
        summary_target = Path(summary_path)
        if summary_target.parent != Path("."):
            summary_target.parent.mkdir(parents=True, exist_ok=True)
        summary_target.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return MicrobenchmarkResult(telemetry_path=str(telemetry_target), summary=summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an offline AutoGen-style prefix reorder microbenchmark.")
    parser.add_argument("--telemetry", required=True, help="JSONL telemetry output path.")
    parser.add_argument("--summary", help="Optional JSON summary output path.")
    parser.add_argument("--session-id", default="microbench", help="Planner session id.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of cold/warm request cycles.")
    parser.add_argument("--agents", nargs="+", default=list(DEFAULT_AGENTS), help="Agent names to simulate.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = asyncio.run(
        run_microbenchmark(
            telemetry_path=args.telemetry,
            summary_path=args.summary,
            agents=tuple(args.agents),
            repeats=args.repeats,
            session_id=args.session_id,
        )
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


class _NoopClient(ChatCompletionClient):
    component_type = "model"

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        return CreateResult(
            finish_reason="stop",
            content="OK",
            usage=RequestUsage(prompt_tokens=len(messages), completion_tokens=1),
            cached=False,
        )

    def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        async def _generator() -> AsyncGenerator[Union[str, CreateResult], None]:
            yield CreateResult(
                finish_reason="stop",
                content="OK",
                usage=RequestUsage(prompt_tokens=len(messages), completion_tokens=1),
                cached=False,
            )

        return _generator()

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def total_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return len(messages)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return 100000 - len(messages)

    @property
    def capabilities(self) -> ModelCapabilities:  # type: ignore
        return {"vision": False, "function_calling": True, "json_output": True}

    @property
    def model_info(self) -> ModelInfo:
        return {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        }


def _messages(agent: str, *, repeat_index: int) -> list[LLMMessage]:
    return [
        SystemMessage(
            content="\n\n".join(
                [
                    "ROLE_SPECIFIC_INSTRUCTION_START\n"
                    f"AGENT_NAME: {agent}\n"
                    f"You are {agent}.\n"
                    "ROLE_SPECIFIC_INSTRUCTION_END",
                    "USER_TASK_START\n"
                    "Build and evaluate the prefix cache experiment.\n"
                    "USER_TASK_END",
                    "SHARED_GROUPCHAT_CONTEXT_START\n"
                    "The team is comparing baseline AutoGen prompts against reordered shared-prefix prompts.\n"
                    "Keep the task objective, shared assumptions, and evaluation criteria stable across agents.\n"
                    "SHARED_GROUPCHAT_CONTEXT_END",
                    "TEAM_POLICY_START\n"
                    "Do not move private memory, latest user instructions, tool results, or role identity into shared prefix.\n"
                    "TEAM_POLICY_END",
                    "TOOL_SCHEMA_START\n"
                    "record_metric(name: string, value: number) -> string\n"
                    "summarize_trace(path: string) -> string\n"
                    "TOOL_SCHEMA_END",
                    "CURRENT_TURN_INSTRUCTION_START\n"
                    f"Offline microbenchmark repeat {repeat_index}: produce a concise status update.\n"
                    "CURRENT_TURN_INSTRUCTION_END",
                ]
            )
        )
    ]


if __name__ == "__main__":
    raise SystemExit(main())
