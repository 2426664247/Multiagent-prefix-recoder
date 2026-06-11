from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import autogen_prefix_tree.readiness as readiness
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
    assert api_check.detail == "missing OPENAI_API_KEY/OAI_CONFIG_LIST/project DeepSeek config"
    assert "config/config.txt" in (api_check.next_step or "")


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


def test_readiness_detects_project_deepseek_config_without_printing_key(tmp_path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_dir.joinpath("config.txt").write_text("deepseek=ds-secret-value\ndeepseek-v4-pro\n", encoding="utf-8")

    report = check_evaluation_readiness(
        cwd=tmp_path,
        env={},
        require_api_config=True,
        require_autogenbench=False,
        mode="legacy-proxy",
    )

    api_check = next(check for check in report.checks if check.name == "api_config")
    serialized = str(report.to_dict())
    assert api_check.ok is True
    assert "project DeepSeek config" in api_check.detail
    assert "deepseek-v4-pro" in api_check.detail
    assert "ds-secret-value" not in serialized


def test_autogenbench_readiness_smokes_working_cli(monkeypatch) -> None:
    monkeypatch.setattr(readiness, "_resolve_autogenbench_cli", lambda: Path("autogenbench"))

    def fake_run(args, **kwargs):
        assert args == ["autogenbench", "--help"]
        assert kwargs["timeout"] == 20
        return SimpleNamespace(returncode=0, stdout="AutoGenBench version 0.0.3\nusage: autogenbench COMMAND ARGS\n", stderr="")

    monkeypatch.setattr(readiness.subprocess, "run", fake_run)

    check = readiness._check_autogenbench(required=True)

    assert check.ok is True
    assert "cli smoke ok" in check.detail
    assert "AutoGenBench version 0.0.3" in check.detail


def test_autogenbench_readiness_reports_broken_cli(monkeypatch) -> None:
    monkeypatch.setattr(readiness, "_resolve_autogenbench_cli", lambda: Path("autogenbench"))

    def fake_run(args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="ModuleNotFoundError: No module named 'autogen'\n")

    monkeypatch.setattr(readiness.subprocess, "run", fake_run)

    check = readiness._check_autogenbench(required=True)

    assert check.ok is False
    assert "ModuleNotFoundError" in check.detail
    assert "pyautogen==0.2.35" in (check.next_step or "")


def test_readiness_reports_missing_autogen_ext_openai(monkeypatch, tmp_path) -> None:
    def fake_find_spec(name: str):
        if name == "autogen_ext.models.openai":
            return None
        return object()

    monkeypatch.setattr(readiness.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(readiness, "_check_virtual_environment", lambda root: readiness.ReadinessCheck("python_environment", True, "ok"))
    monkeypatch.setattr(readiness, "_check_docker", lambda: readiness.ReadinessCheck("docker", True, "ok"))
    monkeypatch.setattr(readiness, "_check_autogenbench", lambda *, required: readiness.ReadinessCheck("autogenbench", True, "ok"))

    report = check_evaluation_readiness(
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "present"},
        require_api_config=True,
        require_autogenbench=False,
    )

    check = next(item for item in report.checks if item.name == "autogen_ext_openai")
    assert check.ok is False
    assert "not installed" in check.detail
    assert "autogen-ext[openai]" in (check.next_step or "")


def test_readiness_reports_missing_autogen_prefix_tree_provider_namespace(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(readiness, "_check_virtual_environment", lambda root: readiness.ReadinessCheck("python_environment", True, "ok"))
    monkeypatch.setattr(readiness, "_check_docker", lambda: readiness.ReadinessCheck("docker", True, "ok"))
    monkeypatch.setattr(readiness, "_check_autogenbench", lambda *, required: readiness.ReadinessCheck("autogenbench", True, "ok"))
    monkeypatch.setattr(readiness, "_check_autogen_ext_openai", lambda: readiness.ReadinessCheck("autogen_ext_openai", True, "ok"))

    missing = check_evaluation_readiness(
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "present"},
        require_api_config=True,
        require_autogenbench=False,
    )
    missing_check = next(item for item in missing.checks if item.name == "autogen_prefix_tree_provider_namespace")
    assert missing_check.ok is False
    assert "AUTOGEN_ALLOWED_PROVIDER_NAMESPACES" in missing_check.detail

    present = check_evaluation_readiness(
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "present", "AUTOGEN_ALLOWED_PROVIDER_NAMESPACES": "autogen_prefix_tree"},
        require_api_config=True,
        require_autogenbench=False,
    )
    present_check = next(item for item in present.checks if item.name == "autogen_prefix_tree_provider_namespace")
    assert present_check.ok is True

    dotted = check_evaluation_readiness(
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "present", "AUTOGEN_ALLOWED_PROVIDER_NAMESPACES": "autogen_prefix_tree.client"},
        require_api_config=True,
        require_autogenbench=False,
    )
    dotted_check = next(item for item in dotted.checks if item.name == "autogen_prefix_tree_provider_namespace")
    assert dotted_check.ok is True


def test_legacy_proxy_readiness_does_not_require_component_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(readiness, "_check_virtual_environment", lambda root: readiness.ReadinessCheck("python_environment", True, "ok"))
    monkeypatch.setattr(readiness, "_check_docker", lambda: readiness.ReadinessCheck("docker", True, "ok"))
    monkeypatch.setattr(readiness, "_check_autogenbench", lambda *, required: readiness.ReadinessCheck("autogenbench", True, "ok"))
    monkeypatch.setattr(readiness, "_check_autogen_core", lambda: readiness.ReadinessCheck("autogen_core", True, "ok"))
    monkeypatch.setattr(
        readiness,
        "_check_autogen_ext_openai",
        lambda: (_ for _ in ()).throw(AssertionError("legacy proxy mode should not check autogen_ext")),
    )
    monkeypatch.setattr(
        readiness,
        "_check_allowed_provider_namespace",
        lambda env: (_ for _ in ()).throw(AssertionError("legacy proxy mode should not check component namespace")),
    )

    report = check_evaluation_readiness(
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "present"},
        require_api_config=True,
        require_autogenbench=True,
        mode="legacy-proxy",
    )

    check_names = {check.name for check in report.checks}
    assert report.ready is True
    assert report.mode == "legacy-proxy"
    assert "autogen_ext_openai" not in check_names
    assert "autogen_prefix_tree_provider_namespace" not in check_names
    assert any("agbench_legacy_suite" in command for command in report.suggested_commands)
    assert any("plugin_rule_only" in command for command in report.suggested_commands)
    assert any("plugin_nl_segmentation" in command for command in report.suggested_commands)
    assert any("offline_semantic_suite" in command and "--source-root" in command for command in report.suggested_commands)
    assert any("three_proxy_smoke" in command and "--manifest" in command for command in report.suggested_commands)
    assert any("semantic_guard_proxy_smoke" in command and "--judge-mode reject" in command for command in report.suggested_commands)
    assert any("agbench_legacy_preflight_smokes" in command and "--manifest" in command for command in report.suggested_commands)
    assert any("agbench_legacy_preflight" in command and "--manifest" in command for command in report.suggested_commands)
    assert any("agbench_legacy_collect" in command for command in report.suggested_commands)
    assert not any("run --subsample" in command for command in report.suggested_commands)
    assert not any("Results/human_eval_two_agents" in command for command in report.suggested_commands)
    assert not any("autogen-ext[openai]" in command for command in report.suggested_commands)


def test_readiness_rejects_unknown_mode(tmp_path) -> None:
    try:
        check_evaluation_readiness(cwd=tmp_path, mode="unknown")
    except ValueError as exc:
        assert "unsupported readiness mode" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_readiness_suggests_matching_autogen_ext_version(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(readiness.metadata, "version", lambda name: "0.7.5")
    monkeypatch.setattr(readiness, "_check_virtual_environment", lambda root: readiness.ReadinessCheck("python_environment", True, "ok"))
    monkeypatch.setattr(readiness, "_check_docker", lambda: readiness.ReadinessCheck("docker", True, "ok"))
    monkeypatch.setattr(readiness, "_check_autogenbench", lambda *, required: readiness.ReadinessCheck("autogenbench", True, "ok"))
    monkeypatch.setattr(readiness, "_check_autogen_ext_openai", lambda: readiness.ReadinessCheck("autogen_ext_openai", True, "ok"))

    report = check_evaluation_readiness(
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "present", "AUTOGEN_ALLOWED_PROVIDER_NAMESPACES": "autogen_prefix_tree"},
        require_api_config=True,
        require_autogenbench=False,
    )

    commands = report.suggested_commands
    assert commands[0] == r'.venv\Scripts\python.exe -m pip install "autogen-ext[openai]==0.7.5"'
    assert commands[1] == r'$env:AUTOGEN_ALLOWED_PROVIDER_NAMESPACES = "autogen_prefix_tree"'
