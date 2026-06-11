from __future__ import annotations

import json

import yaml
from autogen_core.models import ChatCompletionClient

from autogen_prefix_tree import PrefixReorderClient, RequestCaptureClient, StaticResponseClient
from autogen_prefix_tree.agbench_config import build_agbench_config, main
from tests.test_request_capture import FakeClient


def test_build_capture_config_loads_as_chat_completion_client(tmp_path) -> None:
    base = {"model_config": FakeClient().dump_component().model_dump(exclude_none=True)}

    generated = build_agbench_config(
        base,
        mode="capture",
        session_id="capture-config",
        capture_log_path=str(tmp_path / "capture.jsonl"),
    )
    loaded = ChatCompletionClient.load_component(generated["model_config"])

    assert isinstance(loaded, RequestCaptureClient)


def test_build_plugin_config_loads_nested_capture_and_prefix_client(tmp_path) -> None:
    base = {"model_config": FakeClient().dump_component().model_dump(exclude_none=True)}

    generated = build_agbench_config(
        base,
        mode="plugin",
        session_id="plugin-config",
        capture_log_path=str(tmp_path / "capture.jsonl"),
        telemetry_log_path=str(tmp_path / "telemetry.jsonl"),
    )
    loaded = ChatCompletionClient.load_component(generated["model_config"])

    assert isinstance(loaded, PrefixReorderClient)
    assert isinstance(loaded.inner_client, RequestCaptureClient)


def test_build_plugin_config_includes_semantic_guard_config(tmp_path) -> None:
    base = {"model_config": FakeClient().dump_component().model_dump(exclude_none=True)}

    generated = build_agbench_config(
        base,
        mode="plugin",
        session_id="plugin-guard-config",
        capture_log_path=str(tmp_path / "capture.jsonl"),
        telemetry_log_path=str(tmp_path / "telemetry.jsonl"),
        semantic_guard_config={
            "provider": "openai-compatible",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "local-semantic-judge",
            "min_confidence": 0.8,
        },
    )
    loaded = ChatCompletionClient.load_component(generated["model_config"])

    assert isinstance(loaded, PrefixReorderClient)
    guard_config = generated["model_config"]["config"]["semantic_guard"]
    assert guard_config["provider"] == "openai-compatible"
    assert guard_config["base_url"] == "http://127.0.0.1:11434/v1"
    assert guard_config["model"] == "local-semantic-judge"
    assert guard_config["min_confidence"] == 0.8


def test_build_plugin_config_can_enable_natural_language_segmentation(tmp_path) -> None:
    base = {"model_config": FakeClient().dump_component().model_dump(exclude_none=True)}

    generated = build_agbench_config(
        base,
        mode="plugin",
        session_id="plugin-nl-config",
        telemetry_log_path=str(tmp_path / "telemetry.jsonl"),
        enable_natural_language_segmentation=True,
    )
    loaded = ChatCompletionClient.load_component(generated["model_config"])

    assert isinstance(loaded, PrefixReorderClient)
    assert loaded.enable_natural_language_segmentation is True
    assert generated["model_config"]["config"]["enable_natural_language_segmentation"] is True


def test_agbench_config_cli_writes_yaml(tmp_path) -> None:
    base_path = tmp_path / "base.yaml"
    output_path = tmp_path / "plugin.yaml"
    base_path.write_text(
        yaml.safe_dump({"model_config": FakeClient().dump_component().model_dump(exclude_none=True)}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--base-config",
            str(base_path),
            "--output",
            str(output_path),
            "--mode",
            "plugin",
            "--session-id",
            "cli-config",
            "--capture-log-path",
            str(tmp_path / "capture.jsonl"),
            "--telemetry-log-path",
            str(tmp_path / "telemetry.jsonl"),
            "--enable-natural-language-segmentation",
        ]
    )

    assert exit_code == 0
    written = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert written["model_config"]["provider"] == "autogen_prefix_tree.client.PrefixReorderClient"
    assert written["model_config"]["config"]["inner_client"]["provider"] == (
        "autogen_prefix_tree.request_capture.RequestCaptureClient"
    )
    assert written["model_config"]["config"]["enable_natural_language_segmentation"] is True
    json.dumps(written)


def test_agbench_config_cli_writes_semantic_guard_yaml(tmp_path) -> None:
    base_path = tmp_path / "base.yaml"
    output_path = tmp_path / "plugin_guard.yaml"
    base_path.write_text(
        yaml.safe_dump({"model_config": FakeClient().dump_component().model_dump(exclude_none=True)}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--base-config",
            str(base_path),
            "--output",
            str(output_path),
            "--mode",
            "plugin",
            "--session-id",
            "cli-guard-config",
            "--telemetry-log-path",
            str(tmp_path / "telemetry.jsonl"),
            "--semantic-guard-base-url",
            "http://127.0.0.1:11434/v1",
            "--semantic-guard-model",
            "local-semantic-judge",
            "--semantic-guard-min-confidence",
            "0.82",
        ]
    )

    assert exit_code == 0
    written = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    semantic_guard = written["model_config"]["config"]["semantic_guard"]
    assert semantic_guard == {
        "provider": "openai-compatible",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "local-semantic-judge",
        "timeout_seconds": 30.0,
        "min_confidence": 0.82,
        "max_message_chars": 12000,
    }
    json.dumps(written)


def test_build_static_capture_config_loads_without_network_client(tmp_path) -> None:
    base = {"model_config": FakeClient().dump_component().model_dump(exclude_none=True)}

    generated = build_agbench_config(
        base,
        mode="static_capture",
        session_id="static-capture-config",
        capture_log_path=str(tmp_path / "capture.jsonl"),
    )
    loaded = ChatCompletionClient.load_component(generated["model_config"])

    assert isinstance(loaded, RequestCaptureClient)
    assert isinstance(loaded.inner_client, StaticResponseClient)
