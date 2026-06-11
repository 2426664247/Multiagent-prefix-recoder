from __future__ import annotations

import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

from autogen_core.models import ChatCompletionClient

from autogen_prefix_tree import CacheUtilityValidator, OpenAICompatibleSemanticGuard, PrefixReorderClient
from tests.test_native_prefix_reorder import FakeClient, _messages
from tests.test_request_capture import FakeClient as ComponentFakeClient


def test_openai_compatible_semantic_guard_allows_validated_rewrite() -> None:
    seen_requests: list[dict[str, Any]] = []
    server = _start_fake_judge(seen_requests, {"passed": True, "reason": "semantic invariants hold", "checks": ["agent_identity"], "confidence": 0.91})
    try:
        guard = OpenAICompatibleSemanticGuard(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="local-semantic-judge",
        )
        inner = FakeClient()
        client = PrefixReorderClient(
            inner,
            session_id="team",
            validator=CacheUtilityValidator(semantic_guard=guard),
        )

        asyncio.run(client.create(_messages("planner")))
        asyncio.run(client.create(_messages("engineer")))
    finally:
        server.shutdown()
        server.server_close()

    assert client.last_validation_report is not None
    assert client.last_validation_report.fallback is False
    assert client.last_validation_report.semantic_guard_report is not None
    assert client.last_validation_report.semantic_guard_report.passed is True
    assert client.last_validation_report.semantic_guard_report.reason == "semantic_invariants_hold"
    assert len(seen_requests) == 1
    request_payload = json.loads(seen_requests[0]["messages"][1]["content"])
    assert request_payload["moved_blocks"]
    assert request_payload["original_messages"][0]["type"] == "SystemMessage"


def test_openai_compatible_semantic_guard_low_confidence_fails_closed() -> None:
    server = _start_fake_judge([], {"passed": True, "reason": "looks ok", "checks": ["latest_instruction"], "confidence": 0.4})
    try:
        guard = OpenAICompatibleSemanticGuard(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="local-semantic-judge",
            min_confidence=0.8,
        )
        inner = FakeClient()
        client = PrefixReorderClient(
            inner,
            session_id="team",
            validator=CacheUtilityValidator(semantic_guard=guard),
        )

        asyncio.run(client.create(_messages("planner")))
        asyncio.run(client.create(_messages("engineer")))
    finally:
        server.shutdown()
        server.server_close()

    assert client.last_validation_report is not None
    assert client.last_validation_report.fallback is True
    assert client.last_validation_report.reason == "semantic_guard_failed:local_judge_low_confidence"
    second_content = inner.create_calls[1]["messages"][0].content
    assert second_content.index("ROLE_SPECIFIC_INSTRUCTION_START") < second_content.index("USER_TASK_START")


def test_prefix_reorder_component_config_can_enable_openai_compatible_semantic_guard() -> None:
    seen_requests: list[dict[str, Any]] = []
    server = _start_fake_judge(seen_requests, {"passed": False, "reason": "agent identity uncertain", "checks": ["agent_identity"], "confidence": 0.62})
    try:
        loaded = ChatCompletionClient.load_component(
            {
                "provider": "autogen_prefix_tree.client.PrefixReorderClient",
                "component_type": "model",
                "config": {
                    "inner_client": ComponentFakeClient().dump_component().model_dump(exclude_none=True),
                    "session_id": "component-semantic-guard",
                    "semantic_guard": {
                        "provider": "openai-compatible",
                        "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                        "model": "local-semantic-judge",
                        "min_confidence": 0.5,
                    },
                },
            }
        )
        assert isinstance(loaded, PrefixReorderClient)

        asyncio.run(loaded.create(_messages("planner")))
        asyncio.run(loaded.create(_messages("engineer")))
    finally:
        server.shutdown()
        server.server_close()

    assert loaded.last_validation_report is not None
    assert loaded.last_validation_report.fallback is True
    assert loaded.last_validation_report.reason == "semantic_guard_failed:agent_identity_uncertain"
    assert len(seen_requests) == 1


def _start_fake_judge(seen_requests: list[dict[str, Any]], judge_result: dict[str, Any]) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or "0")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            seen_requests.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(judge_result),
                                }
                            }
                        ]
                    }
                ).encode("utf-8")
            )

        def log_message(self, format, *args) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
