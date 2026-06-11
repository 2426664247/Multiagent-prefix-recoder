from __future__ import annotations

import json

from autogen_prefix_tree.semantic_guard_proxy_smoke import main, run_semantic_guard_proxy_smoke


def test_semantic_guard_proxy_smoke_reject_mode_fails_closed(tmp_path) -> None:
    result = run_semantic_guard_proxy_smoke(
        output_dir=tmp_path / "reject",
        session_id="semantic-guard-reject-smoke",
        judge_mode="reject",
    )

    summary = result.summary
    assert summary["schema_version"] == "prefix-semantic-guard-proxy-smoke-summary-v1"
    assert summary["fake_upstream"] is True
    assert summary["fake_local_judge"] is True
    assert summary["real_provider_metrics_available"] is False
    assert summary["real_semantic_model_metrics_available"] is False
    assert summary["request_count"] == 2
    assert summary["judge_request_count"] == 1
    assert summary["upstream_request_count"] == 2
    assert summary["adapter_validation_reasons"] == [
        "no_rewrite_needed",
        "semantic_guard_failed:private_boundary_uncertain",
    ]
    assert summary["provider_rewrite_applied"] == [False, False]
    assert summary["warm_provider_rewrite_applied"] is False
    assert summary["warm_semantic_guard"]["passed"] is False
    assert summary["warm_semantic_guard"]["reason"] == "private_boundary_uncertain"
    assert summary["provider_summary"]["record_count"] == 2
    assert "Build the cache experiment" not in json.dumps(summary, ensure_ascii=False)


def test_semantic_guard_proxy_smoke_accept_mode_allows_rewrite(tmp_path) -> None:
    result = run_semantic_guard_proxy_smoke(
        output_dir=tmp_path / "accept",
        session_id="semantic-guard-accept-smoke",
        judge_mode="accept",
    )

    summary = result.summary
    assert summary["adapter_validation_reasons"] == ["no_rewrite_needed", "validated"]
    assert summary["provider_rewrite_applied"] == [False, True]
    assert summary["warm_provider_rewrite_applied"] is True
    assert summary["warm_semantic_guard"]["passed"] is True
    assert summary["warm_semantic_guard"]["confidence"] == 0.93


def test_semantic_guard_proxy_smoke_cli_writes_summary(tmp_path) -> None:
    output_dir = tmp_path / "cli"

    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--session-id",
            "semantic-guard-cli-smoke",
            "--judge-mode",
            "low-confidence",
            "--min-confidence",
            "0.8",
        ]
    )

    summary = json.loads((output_dir / "semantic_guard_proxy_smoke_summary.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["warm_validation"]["reason"] == "semantic_guard_failed:local_judge_low_confidence"
    assert summary["warm_semantic_guard"]["passed"] is False
    assert summary["warm_semantic_guard"]["confidence"] == 0.42
    assert summary["warm_provider_rewrite_applied"] is False
