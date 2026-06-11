from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from .dataset_eval import evaluate_dataset
from .request_capture import RequestCaptureClient
from .static_client import StaticResponseClient


MAGENTIC_ONE_LIKE_CODER_SYSTEM = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code in a python coding block for the user to execute.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible."""


@dataclass(frozen=True)
class AutoGenCaptureSmokeResult:
    capture_path: str
    summary_path: str
    telemetry_path: str
    summary: dict[str, Any]
    message_count: int
    stop_reason: str | None


async def run_autogen_capture_smoke(
    *,
    capture_path: str | Path,
    summary_path: str | Path,
    telemetry_path: str | Path,
    session_id: str = "autogen-capture-smoke",
    prompt_style: str = "marked",
    task: str = "Complete a HumanEval-style add(a, b) programming task. Reply TERMINATE.",
    max_messages: int = 3,
) -> AutoGenCaptureSmokeResult:
    capture_target = Path(capture_path)
    summary_target = Path(summary_path)
    telemetry_target = Path(telemetry_path)
    for target in (capture_target, summary_target, telemetry_target):
        if target.exists():
            target.unlink()

    model_client = RequestCaptureClient(
        StaticResponseClient(responses=("TERMINATE",)),
        capture_log_path=capture_target,
        session_id=session_id,
    )
    planner = AssistantAgent(
        "planner",
        model_client=model_client,
        system_message=_system_message("planner", "Plan the solution.", style=prompt_style),
    )
    engineer = AssistantAgent(
        "engineer",
        model_client=model_client,
        system_message=_system_message("engineer", "Implement the solution.", style=prompt_style),
    )
    team = RoundRobinGroupChat(
        [planner, engineer],
        termination_condition=MaxMessageTermination(max_messages=max_messages),
    )
    result = await team.run(task=task)
    await model_client.close()

    evaluation = evaluate_dataset(
        input_path=capture_target,
        telemetry_path=telemetry_target,
        summary_path=summary_target,
        session_id=session_id,
    )
    return AutoGenCaptureSmokeResult(
        capture_path=str(capture_target),
        summary_path=str(summary_target),
        telemetry_path=str(telemetry_target),
        summary=evaluation.summary,
        message_count=len(result.messages),
        stop_reason=result.stop_reason,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an offline AutoGen AgentChat capture and prefix coverage smoke.")
    parser.add_argument("--capture", required=True, help="Output AutoGen typed request capture JSONL.")
    parser.add_argument("--summary", required=True, help="Output dataset_eval summary JSON.")
    parser.add_argument("--telemetry", required=True, help="Output prompt-safe dataset_eval telemetry JSONL.")
    parser.add_argument("--session-id", default="autogen-capture-smoke")
    parser.add_argument("--prompt-style", choices=("marked", "magentic_one_like"), default="marked")
    parser.add_argument("--task", default="Complete a HumanEval-style add(a, b) programming task. Reply TERMINATE.")
    parser.add_argument("--max-messages", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = asyncio.run(
        run_autogen_capture_smoke(
            capture_path=args.capture,
            summary_path=args.summary,
            telemetry_path=args.telemetry,
            session_id=args.session_id,
            prompt_style=args.prompt_style,
            task=args.task,
            max_messages=args.max_messages,
        )
    )
    print(
        json.dumps(
            {
                "capture_path": result.capture_path,
                "summary_path": result.summary_path,
                "telemetry_path": result.telemetry_path,
                "message_count": result.message_count,
                "stop_reason": result.stop_reason,
                "summary": result.summary,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _system_message(agent_name: str, role_text: str, *, style: str) -> str:
    if style == "marked":
        return "\n\n".join(
            [
                "ROLE_SPECIFIC_INSTRUCTION_START\n"
                f"AGENT_NAME: {agent_name}\n"
                f"{role_text}\n"
                "ROLE_SPECIFIC_INSTRUCTION_END",
                "USER_TASK_START\nComplete a HumanEval-style programming task.\nUSER_TASK_END",
                "SHARED_GROUPCHAT_CONTEXT_START\n"
                "The team is solving the same programming task and checking cache-prefix behavior.\n"
                "SHARED_GROUPCHAT_CONTEXT_END",
                "TEAM_POLICY_START\nDo not move private memory, tool results, or latest user instructions.\nTEAM_POLICY_END",
                "CURRENT_TURN_INSTRUCTION_START\nReply with TERMINATE for this offline smoke.\nCURRENT_TURN_INSTRUCTION_END",
            ]
        )
    if style == "magentic_one_like":
        return MAGENTIC_ONE_LIKE_CODER_SYSTEM
    raise ValueError(f"unsupported prompt style: {style}")


if __name__ == "__main__":
    raise SystemExit(main())
