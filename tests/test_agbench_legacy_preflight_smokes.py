from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import autogen_prefix_tree.agbench_legacy_preflight as preflight
from autogen_prefix_tree.agbench_legacy_preflight_smokes import main, run_legacy_preflight_smokes
from autogen_prefix_tree.agbench_legacy_suite import generate_legacy_agbench_suite


def test_legacy_preflight_smokes_fills_manifest_artifacts_and_refreshes_reports(monkeypatch, tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-preflight-smokes",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    result = run_legacy_preflight_smokes(
        manifest_path=suite.manifest_path,
        output_dir=tmp_path / "legacy" / "preflight_smokes",
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
    )

    summary = result.summary
    manifest = suite.manifest
    assert summary["schema_version"] == "prefix-legacy-preflight-smokes-summary-v1"
    assert summary["fake_upstream"] is True
    assert summary["fake_local_judge"] is True
    assert summary["local_judge_healthcheck_run"] is False
    assert summary["local_judge_healthcheck_ready"] is False
    assert summary["fake_local_judge_matrix_smoke_run"] is False
    assert summary["offline_local_judge_matrix_smoke_ready"] is False
    assert summary["real_provider_metrics_available"] is False
    assert summary["provider_or_ab_fake_artifacts_detected"] is True
    assert summary["semantic_guard_fake_smoke_ready"] is True
    assert summary["next_action"] == "run_offline_semantic_suite"
    assert summary["ready_to_claim_real_results"] is False
    assert "fake judge" in summary["real_provider_metrics_note"]

    three_summary = json.loads(open(summary["three_proxy_smoke_summary_path"], encoding="utf-8").read())
    semantic_summary = json.loads(open(summary["semantic_guard_smoke_summary_path"], encoding="utf-8").read())
    runbook_summary = json.loads(open(summary["legacy_runbook_summary_path"], encoding="utf-8").read())
    preflight_summary = json.loads(open(summary["legacy_real_ab_preflight_summary_path"], encoding="utf-8").read())
    assert three_summary["fake_upstream"] is True
    assert semantic_summary["fake_local_judge"] is True
    assert semantic_summary["judge_mode"] == "reject"
    assert runbook_summary["provider_or_ab_fake_artifacts_detected"] is True
    assert runbook_summary["semantic_guard_fake_smoke_ready"] is True
    assert preflight_summary["gates"]["fake_artifacts_detected"] is True
    assert preflight_summary["gates"]["semantic_guard_fake_smoke_ready"] is True
    assert preflight_summary["gates"]["offline_local_judge_matrix_smoke_ready"] is False
    assert preflight_summary["next_action"] == "run_offline_semantic_suite"
    assert preflight_summary["gates"]["ready_to_claim_real_results"] is False

    baseline_provider = _read_jsonl(manifest["oai_config_lists"]["baseline"]["provider_telemetry_path"])
    assert baseline_provider
    assert all(row["fake_upstream"] is True for row in baseline_provider)


def test_legacy_preflight_smokes_cli_writes_summary(monkeypatch, tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-preflight-smokes-cli",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)
    summary_path = tmp_path / "legacy" / "reports" / "preflight_smokes_summary.json"

    exit_code = main(
        [
            "--manifest",
            suite.manifest_path,
            "--output-dir",
            str(tmp_path / "legacy" / "preflight_smokes"),
            "--summary",
            str(summary_path),
            "--cwd",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["suite_id"] == "legacy-preflight-smokes-cli"
    assert summary["semantic_guard_fake_smoke_ready"] is True
    assert summary["next_action"] == "run_offline_semantic_suite"


def test_legacy_preflight_smokes_can_run_local_judge_healthcheck(monkeypatch, tmp_path) -> None:
    seen_requests: list[dict] = []
    judge = _start_fake_openai_compatible_judge(seen_requests)
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-preflight-smokes-local-healthcheck",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url=f"http://127.0.0.1:{judge.server_port}/v1",
        local_judge_model="local-healthcheck-model",
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    try:
        result = run_legacy_preflight_smokes(
            manifest_path=suite.manifest_path,
            output_dir=tmp_path / "legacy" / "preflight_smokes",
            cwd=tmp_path,
            env={"OPENAI_API_KEY": "sk-secret-value"},
            run_real_local_judge_healthcheck=True,
        )
    finally:
        judge.shutdown()
        judge.server_close()

    summary = result.summary
    healthcheck_summary = json.loads(open(summary["local_judge_healthcheck_summary_path"], encoding="utf-8").read())
    runbook_summary = json.loads(open(summary["legacy_runbook_summary_path"], encoding="utf-8").read())
    preflight_summary = json.loads(open(summary["legacy_real_ab_preflight_summary_path"], encoding="utf-8").read())
    assert summary["local_judge_healthcheck_run"] is True
    assert summary["local_judge_healthcheck_ready"] is True
    assert summary["local_judge_healthcheck_runbook_ready"] is True
    assert healthcheck_summary["ready"] is True
    assert healthcheck_summary["model"] == "local-healthcheck-model"
    assert runbook_summary["local_judge_healthcheck_ready"] is True
    assert preflight_summary["gates"]["local_judge_healthcheck_ready"] is True
    assert len(seen_requests) == 1
    assert seen_requests[0]["model"] == "local-healthcheck-model"


def test_legacy_preflight_smokes_can_run_real_local_judge_matrix(monkeypatch, tmp_path) -> None:
    seen_requests: list[dict] = []
    judge = _start_fake_openai_compatible_judge(seen_requests)
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-preflight-smokes-real-local-matrix",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url=f"http://127.0.0.1:{judge.server_port}/v1",
        local_judge_model="local-healthcheck-model",
    )
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "agent.py").write_text(
        '''
SYSTEM_MESSAGE = """
You are a helpful AI assistant.

When using code, indicate the script type in the code block.

Select the next step in the plan before responding.

Verify the answer carefully and include evidence.
"""
''',
        encoding="utf-8",
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    try:
        result = run_legacy_preflight_smokes(
            manifest_path=suite.manifest_path,
            output_dir=tmp_path / "legacy" / "preflight_smokes",
            cwd=tmp_path,
            env={"OPENAI_API_KEY": "sk-secret-value"},
            local_judge_sources=[("framework", source_root)],
            run_real_local_judge_matrix=True,
            max_local_judge_calls=1,
        )
    finally:
        judge.shutdown()
        judge.server_close()

    summary = result.summary
    runbook_summary = json.loads(open(summary["legacy_runbook_summary_path"], encoding="utf-8").read())
    preflight_summary = json.loads(open(summary["legacy_real_ab_preflight_summary_path"], encoding="utf-8").read())
    matrix_summary = json.loads(open(summary["offline_local_judge_matrix_summary_path"], encoding="utf-8").read())
    assert summary["real_local_judge_matrix_run"] is True
    assert summary["offline_local_judge_matrix_ready"] is True
    assert matrix_summary["judge_mode"] == "openai-compatible"
    assert matrix_summary["include_candidate_text"] is True
    assert matrix_summary["aggregate"]["local_judge_action_counts"]["model_called"] > 0
    assert runbook_summary["offline_local_judge_matrix_ready"] is True
    assert preflight_summary["gates"]["offline_local_judge_matrix_ready"] is True
    assert len(seen_requests) >= 2


def test_legacy_preflight_smokes_can_run_local_judge_matrix_smoke(monkeypatch, tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-preflight-smokes-local-judge",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "agent.py").write_text(
        '''
SYSTEM_MESSAGE = """
You are a helpful AI assistant.

When using code, indicate the script type in the code block.

Select the next step in the plan before responding.

Verify the answer carefully and include evidence.
"""
''',
        encoding="utf-8",
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    result = run_legacy_preflight_smokes(
        manifest_path=suite.manifest_path,
        output_dir=tmp_path / "legacy" / "preflight_smokes",
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        local_judge_sources=[("framework", source_root)],
        max_local_judge_calls=1,
    )

    summary = result.summary
    runbook_summary = json.loads(open(summary["legacy_runbook_summary_path"], encoding="utf-8").read())
    preflight_summary = json.loads(open(summary["legacy_real_ab_preflight_summary_path"], encoding="utf-8").read())
    matrix_summary = json.loads(open(summary["offline_local_judge_matrix_smoke_summary_path"], encoding="utf-8").read())
    assert summary["fake_local_judge_matrix_smoke_run"] is True
    assert summary["offline_local_judge_matrix_smoke_ready"] is True
    assert matrix_summary["fake_local_judge"] is True
    assert matrix_summary["fake_local_judge_request_count"] > 0
    assert runbook_summary["offline_local_judge_matrix_smoke_ready"] is True
    assert preflight_summary["gates"]["offline_local_judge_matrix_smoke_ready"] is True


def _patch_readiness(monkeypatch, *, ready: bool, api_present: bool) -> None:
    detail = "present: OPENAI_API_KEY" if api_present else "missing OPENAI_API_KEY/OAI_CONFIG_LIST"
    report = {
        "ready": ready,
        "mode": "legacy-proxy",
        "checks": [
            {"name": "python_environment", "ok": True, "detail": "ok"},
            {"name": "docker", "ok": True, "detail": "ok"},
            {"name": "api_config", "ok": api_present or ready, "detail": detail},
            {"name": "autogenbench", "ok": True, "detail": "ok"},
            {"name": "autogen_core", "ok": True, "detail": "ok"},
        ],
        "suggested_commands": ["readiness-command"],
    }
    monkeypatch.setattr(preflight, "check_evaluation_readiness", lambda **kwargs: SimpleNamespace(to_dict=lambda: report))


def _task_jsonl(tmp_path):
    tasks_dir = tmp_path / "Tasks"
    tasks_dir.mkdir(exist_ok=True)
    template = tasks_dir / "template.py"
    template.write_text("print('template')\n", encoding="utf-8")
    path = tasks_dir / "human_eval_two_agents.jsonl"
    path.write_text(
        json.dumps({"id": "HumanEval_0", "template": "template.py", "substitutions": {}}) + "\n",
        encoding="utf-8",
    )
    return path


def _read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def _start_fake_openai_compatible_judge(seen_requests: list[dict]) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or "0")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            seen_requests.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "label": "accept",
                                    "reason": "stable verification policy",
                                    "confidence": 0.91,
                                }
                            )
                        }
                    }
                ]
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))

        def log_message(self, format, *args) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
