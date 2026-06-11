from __future__ import annotations

import json

import yaml
from autogen_core.models import ChatCompletionClient

from autogen_prefix_tree import PrefixReorderClient, RequestCaptureClient, StaticResponseClient
from autogen_prefix_tree.agbench_suite import generate_agbench_suite, main
from tests.test_request_capture import FakeClient


def test_generate_agbench_suite_writes_loadable_configs_and_manifest(tmp_path) -> None:
    base_path = tmp_path / "base.yaml"
    output_dir = tmp_path / "suite"
    base_path.write_text(
        yaml.safe_dump({"model_config": FakeClient().dump_component().model_dump(exclude_none=True)}),
        encoding="utf-8",
    )

    result = generate_agbench_suite(
        base_config_path=base_path,
        output_dir=output_dir,
        suite_id="suite-smoke",
    )

    assert result.manifest_path == str(output_dir / "suite_manifest.json")
    assert result.manifest["schema_version"] == "prefix-agbench-suite-manifest-v1"
    assert set(result.manifest["config_variants"]) == {
        "baseline",
        "capture",
        "plugin_rule_only",
        "plugin_nl_segmentation",
        "static_capture",
    }

    baseline = ChatCompletionClient.load_component(_load_model_config(output_dir / "baseline.yaml"))
    capture = ChatCompletionClient.load_component(_load_model_config(output_dir / "capture.yaml"))
    rule = ChatCompletionClient.load_component(_load_model_config(output_dir / "plugin_rule_only.yaml"))
    nl = ChatCompletionClient.load_component(_load_model_config(output_dir / "plugin_nl_segmentation.yaml"))
    static_capture = ChatCompletionClient.load_component(_load_model_config(output_dir / "static_capture.yaml"))

    assert isinstance(baseline, FakeClient)
    assert isinstance(capture, RequestCaptureClient)
    assert isinstance(rule, PrefixReorderClient)
    assert isinstance(rule.inner_client, RequestCaptureClient)
    assert isinstance(nl, PrefixReorderClient)
    assert isinstance(nl.inner_client, RequestCaptureClient)
    assert isinstance(static_capture, RequestCaptureClient)
    assert isinstance(static_capture.inner_client, StaticResponseClient)
    assert rule.enable_natural_language_segmentation is False
    assert nl.enable_natural_language_segmentation is True
    assert "offline_semantic_suite" in result.manifest["suggested_commands"]
    assert "offline_semantic_suite" in result.manifest["suggested_commands"]["offline_semantic_suite"][0]
    assert "--source-root <autogen-source-root>" in result.manifest["suggested_commands"]["offline_semantic_suite"][0]


def test_generate_agbench_suite_can_include_semantic_guard(tmp_path) -> None:
    base_path = tmp_path / "base.yaml"
    output_dir = tmp_path / "suite"
    base_path.write_text(
        yaml.safe_dump({"model_config": FakeClient().dump_component().model_dump(exclude_none=True)}),
        encoding="utf-8",
    )

    result = generate_agbench_suite(
        base_config_path=base_path,
        output_dir=output_dir,
        suite_id="suite-guard",
        include_guard=True,
        semantic_guard_config={
            "provider": "openai-compatible",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "local-semantic-judge",
            "min_confidence": 0.81,
        },
    )

    guard_config = yaml.safe_load((output_dir / "plugin_guard.yaml").read_text(encoding="utf-8"))
    guard_model_config = guard_config["model_config"]["config"]
    assert guard_model_config["enable_natural_language_segmentation"] is True
    assert guard_model_config["semantic_guard"]["provider"] == "openai-compatible"
    assert result.manifest["config_variants"]["plugin_guard"]["semantic_guard_enabled"] is True
    assert result.manifest["recommended_comparisons"][-1] == {
        "baseline": "baseline",
        "candidate": "plugin_guard",
    }


def test_agbench_suite_manifest_is_prompt_safe(tmp_path) -> None:
    base_path = tmp_path / "secret_base.yaml"
    output_dir = tmp_path / "suite"
    base_path.write_text(
        yaml.safe_dump(
            {
                "model_config": FakeClient().dump_component().model_dump(exclude_none=True),
                "private_prompt": "SHOULD_NOT_APPEAR_IN_MANIFEST",
                "api_key": "sk-should-not-appear",
            }
        ),
        encoding="utf-8",
    )

    result = generate_agbench_suite(
        base_config_path=base_path,
        output_dir=output_dir,
        suite_id="suite-redacted",
        include_message_content=False,
    )

    serialized = json.dumps(result.manifest, ensure_ascii=False)
    assert result.manifest["prompt_safe_manifest"] is True
    assert result.manifest["capture_files_may_contain_prompt_text"] is False
    assert "SHOULD_NOT_APPEAR_IN_MANIFEST" not in serialized
    assert "sk-should-not-appear" not in serialized
    assert "autogen_prefix_tree" in serialized


def test_agbench_suite_cli_writes_guard_suite(tmp_path) -> None:
    base_path = tmp_path / "base.yaml"
    output_dir = tmp_path / "suite"
    base_path.write_text(
        yaml.safe_dump({"model_config": FakeClient().dump_component().model_dump(exclude_none=True)}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--base-config",
            str(base_path),
            "--output-dir",
            str(output_dir),
            "--suite-id",
            "cli-suite",
            "--redact-message-content",
            "--include-guard",
            "--semantic-guard-base-url",
            "http://127.0.0.1:11434/v1",
            "--semantic-guard-model",
            "local-semantic-judge",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_dir / "suite_manifest.json").read_text(encoding="utf-8"))
    assert "plugin_guard" in manifest["config_variants"]
    assert manifest["capture_files_may_contain_prompt_text"] is False
    assert (output_dir / "plugin_guard.yaml").exists()


def _load_model_config(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))["model_config"]
