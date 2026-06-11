from __future__ import annotations

import json
from pathlib import Path

import yaml

from autogen_prefix_tree.agbench_suite import generate_agbench_suite
from autogen_prefix_tree.agbench_suite_verify import main, verify_agbench_suite
from tests.test_request_capture import FakeClient


def test_verify_agbench_suite_accepts_fresh_generated_suite(tmp_path) -> None:
    result = _generate_suite(tmp_path)

    verified = verify_agbench_suite(manifest_path=result.manifest_path)

    assert verified.summary["ready"] is True
    assert verified.summary["schema_version"] == "prefix-agbench-suite-verification-v1"
    assert verified.summary["real_provider_metrics_available"] is False
    assert _check(verified.summary, "config:plugin_rule_only:natural_language_segmentation")["ok"] is True
    assert _check(verified.summary, "config:plugin_nl_segmentation:natural_language_segmentation")["ok"] is True
    assert _check(verified.summary, "config:static_capture:inner_static_client")["ok"] is True


def test_verify_agbench_suite_reports_missing_required_artifacts(tmp_path) -> None:
    result = _generate_suite(tmp_path)

    verified = verify_agbench_suite(
        manifest_path=result.manifest_path,
        require_provider_artifacts=True,
        require_task_artifacts=True,
    )

    assert verified.summary["ready"] is False
    failed = {check["name"] for check in verified.summary["checks"] if not check["ok"]}
    assert "artifact:baseline:provider_telemetry_path" in failed
    assert "artifact:plugin_rule_only:task_results_path" in failed


def test_verify_agbench_suite_can_require_existing_capture_artifacts(tmp_path) -> None:
    result = _generate_suite(tmp_path)
    manifest = result.manifest
    for variant in manifest["config_variants"].values():
        capture_path = variant.get("capture_log_path")
        if capture_path:
            target = Path(capture_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}\n", encoding="utf-8")

    verified = verify_agbench_suite(
        manifest_path=result.manifest_path,
        require_capture_artifacts=True,
    )

    assert verified.summary["ready"] is True
    capture_checks = [check for check in verified.summary["checks"] if check["name"].startswith("artifact:")]
    assert capture_checks
    assert all(check["ok"] for check in capture_checks)


def test_verify_agbench_suite_rejects_manifest_with_embedded_model_config(tmp_path) -> None:
    result = _generate_suite(tmp_path)
    manifest_path = tmp_path / "suite" / "suite_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_variants"]["baseline"]["model_config"] = {"api_key": "sk-should-not-appear"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    verified = verify_agbench_suite(manifest_path=manifest_path, load_components=False)

    assert verified.summary["ready"] is False
    failed = {check["name"] for check in verified.summary["checks"] if not check["ok"]}
    assert "manifest:no_embedded_model_config" in failed


def test_agbench_suite_verify_cli_writes_summary_and_uses_exit_code(tmp_path) -> None:
    result = _generate_suite(tmp_path)
    summary_path = tmp_path / "verify_summary.json"

    exit_code = main(["--manifest", result.manifest_path, "--summary", str(summary_path)])

    assert exit_code == 0
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["ready"] is True
    assert written["prompt_safe_summary"] is True


def _generate_suite(tmp_path):
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump({"model_config": FakeClient().dump_component().model_dump(exclude_none=True)}),
        encoding="utf-8",
    )
    return generate_agbench_suite(
        base_config_path=base_path,
        output_dir=tmp_path / "suite",
        suite_id="verify-suite",
        include_guard=True,
        semantic_guard_config={
            "provider": "openai-compatible",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "local-semantic-judge",
        },
    )


def _check(summary, name: str):
    return next(check for check in summary["checks"] if check["name"] == name)
