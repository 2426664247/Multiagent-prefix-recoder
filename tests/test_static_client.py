from __future__ import annotations

import asyncio
import json

from autogen_core.models import ChatCompletionClient, SystemMessage

from autogen_prefix_tree import RequestCaptureClient, StaticResponseClient
from autogen_prefix_tree.dataset_eval import evaluate_dataset


def test_static_response_client_loads_from_component_config() -> None:
    loaded = ChatCompletionClient.load_component(
        {
            "provider": "autogen_prefix_tree.static_client.StaticResponseClient",
            "component_type": "model",
            "config": {
                "responses": ["first", "second"],
                "model": "offline",
            },
        }
    )

    assert isinstance(loaded, StaticResponseClient)
    first = asyncio.run(loaded.create([SystemMessage(content="hello")]))
    second = asyncio.run(loaded.create([SystemMessage(content="hello")]))
    assert first.content == "first"
    assert second.content == "second"


def test_static_response_client_can_drive_capture_and_dataset_eval(tmp_path) -> None:
    capture_path = tmp_path / "captured.jsonl"
    client = RequestCaptureClient(
        StaticResponseClient(responses=["TERMINATE"]),
        capture_log_path=capture_path,
        session_id="static-capture",
    )

    asyncio.run(client.create([SystemMessage(content=_system("planner"))]))
    asyncio.run(client.create([SystemMessage(content=_system("engineer"))]))

    rows = [json.loads(line) for line in capture_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["body"]["messages"][0]["type"] == "SystemMessage"
    result = evaluate_dataset(input_path=capture_path, session_id="static-capture")
    assert result.summary["supported_request_count"] == 2
    assert result.summary["validation_reason_counts"] == {"no_rewrite_needed": 1, "validated": 1}


def _system(agent: str) -> str:
    return "\n\n".join(
        [
            "ROLE_SPECIFIC_INSTRUCTION_START\n"
            f"AGENT_NAME: {agent}\n"
            f"You are {agent}.\n"
            "ROLE_SPECIFIC_INSTRUCTION_END",
            "USER_TASK_START\nBuild the cache experiment.\nUSER_TASK_END",
            "SHARED_GROUPCHAT_CONTEXT_START\nShared benchmark context.\nSHARED_GROUPCHAT_CONTEXT_END",
            "TEAM_POLICY_START\nDo not move private memory or latest user instructions.\nTEAM_POLICY_END",
            "CURRENT_TURN_INSTRUCTION_START\nReport status.\nCURRENT_TURN_INSTRUCTION_END",
        ]
    )
