from __future__ import annotations

from autogen_prefix_tree.readiness import check_evaluation_readiness


def test_readiness_reports_missing_api_without_secret_values(tmp_path) -> None:
    report = check_evaluation_readiness(
        cwd=tmp_path,
        env={},
        require_api_config=True,
        require_autogenbench=False,
    )

    api_check = next(check for check in report.checks if check.name == "api_config")
    assert api_check.ok is False
    assert api_check.detail == "missing OPENAI_API_KEY/OAI_CONFIG_LIST"
    assert "Set OPENAI_API_KEY" in (api_check.next_step or "")


def test_readiness_detects_api_markers_without_printing_values(tmp_path) -> None:
    (tmp_path / "OAI_CONFIG_LIST").write_text("secret-file-content", encoding="utf-8")
    report = check_evaluation_readiness(
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value", "OAI_CONFIG_LIST": "secret-json"},
        require_api_config=True,
        require_autogenbench=False,
    )

    api_check = next(check for check in report.checks if check.name == "api_config")
    serialized = str(report.to_dict())
    assert api_check.ok is True
    assert "OPENAI_API_KEY" in api_check.detail
    assert "OAI_CONFIG_LIST env" in api_check.detail
    assert "OAI_CONFIG_LIST file" in api_check.detail
    assert "sk-secret-value" not in serialized
    assert "secret-json" not in serialized
    assert "secret-file-content" not in serialized
