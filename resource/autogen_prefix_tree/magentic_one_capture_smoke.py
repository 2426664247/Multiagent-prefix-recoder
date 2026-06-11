from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from .dataset_eval import evaluate_dataset
from .request_capture import RequestCaptureClient
from .static_client import StaticResponseClient


@dataclass(frozen=True)
class MagenticOneCaptureSmokeResult:
    capture_path: str
    summary_path: str
    telemetry_path: str
    summary: dict[str, Any]
    message_count: int
    stop_reason: str | None


async def run_magentic_one_coder_capture_smoke(
    *,
    capture_path: str | Path,
    summary_path: str | Path,
    telemetry_path: str | Path,
    session_id: str = "magentic-one-coder-static",
    task: str = "Complete def add(a, b): return a + b. Reply TERMINATE.",
    max_messages: int = 3,
    agent_cls: Any | None = None,
) -> MagenticOneCaptureSmokeResult:
    capture_target = Path(capture_path)
    summary_target = Path(summary_path)
    telemetry_target = Path(telemetry_path)
    for target in (capture_target, summary_target, telemetry_target):
        if target.exists():
            target.unlink()

    coder_cls = agent_cls or _load_magentic_one_coder_agent()
    model_client = RequestCaptureClient(
        StaticResponseClient(
            responses=(
                "```python\ndef add(a, b):\n    return a + b\n```",
                "```python\ndef add(a, b):\n    return a + b\n```\nTERMINATE",
            )
        ),
        capture_log_path=capture_target,
        session_id=session_id,
    )
    try:
        coder = coder_cls(name="coder", model_client=model_client)
        team = RoundRobinGroupChat(
            [coder],
            termination_condition=MaxMessageTermination(max_messages=max_messages),
        )
        result = await team.run(task=task)
    finally:
        await model_client.close()

    evaluation = evaluate_dataset(
        input_path=capture_target,
        telemetry_path=telemetry_target,
        summary_path=summary_target,
        session_id=session_id,
    )
    return MagenticOneCaptureSmokeResult(
        capture_path=str(capture_target),
        summary_path=str(summary_target),
        telemetry_path=str(telemetry_target),
        summary=evaluation.summary,
        message_count=len(result.messages),
        stop_reason=result.stop_reason,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run no-network MagenticOneCoderAgent request capture and prefix coverage smoke."
    )
    parser.add_argument("--capture", required=True, help="Output AutoGen typed request capture JSONL.")
    parser.add_argument("--summary", required=True, help="Output dataset_eval summary JSON.")
    parser.add_argument("--telemetry", required=True, help="Output prompt-safe dataset_eval telemetry JSONL.")
    parser.add_argument("--session-id", default="magentic-one-coder-static")
    parser.add_argument("--task", default="Complete def add(a, b): return a + b. Reply TERMINATE.")
    parser.add_argument("--max-messages", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = asyncio.run(
        run_magentic_one_coder_capture_smoke(
            capture_path=args.capture,
            summary_path=args.summary,
            telemetry_path=args.telemetry,
            session_id=args.session_id,
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


def _load_magentic_one_coder_agent() -> Any:
    try:
        from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "autogen_ext.agents.magentic_one is not installed in this interpreter; "
            "run with the AutoGen source environment or install the matching autogen-ext package."
        ) from exc
    return MagenticOneCoderAgent


if __name__ == "__main__":
    raise SystemExit(main())
