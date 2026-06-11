from __future__ import annotations

import asyncio
from typing import Any

from autogen_agentchat.agents import AssistantAgent

from autogen_prefix_tree.magentic_one_capture_smoke import run_magentic_one_coder_capture_smoke


MAGENTIC_ONE_STYLE_SYSTEM = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code in a python coding block for the user to execute.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible."""


def test_magentic_one_capture_smoke_exposes_no_shared_prefix_with_natural_prompt(tmp_path) -> None:
    def fake_magentic_one_agent(*, name: str, model_client: Any) -> AssistantAgent:
        return AssistantAgent(name=name, model_client=model_client, system_message=MAGENTIC_ONE_STYLE_SYSTEM)

    result = asyncio.run(
        run_magentic_one_coder_capture_smoke(
            capture_path=tmp_path / "messages.jsonl",
            summary_path=tmp_path / "summary.json",
            telemetry_path=tmp_path / "telemetry.jsonl",
            session_id="fake-magentic-one",
            agent_cls=fake_magentic_one_agent,
        )
    )

    assert result.summary["supported_request_count"] == 2
    assert result.summary["applied_count"] == 0
    assert result.summary["reusable_prefix_request_count"] == 0
    assert result.summary["validation_reason_counts"] == {"no_rewrite_needed": 2}
    assert result.summary["semantic_type_counts"]["role_identity"] == 2
    assert "shared_context" not in result.summary["semantic_type_counts"]
    assert "team_policy" not in result.summary["semantic_type_counts"]
