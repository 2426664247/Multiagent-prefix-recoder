from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

from autogen_prefix_tree.agbench_legacy_native_run import main, run_legacy_native_ab_sample
from autogen_prefix_tree.agbench_legacy_suite import generate_legacy_agbench_suite


def test_legacy_native_run_executes_fixed_sample_and_collects_metrics(tmp_path) -> None:
    upstream_requests: list[dict[str, Any]] = []
    upstream = _start_fake_upstream(upstream_requests)
    try:
        suite = generate_legacy_agbench_suite(
            output_dir=tmp_path / "legacy",
            suite_id="native-smoke",
            model="fake-model",
            upstream_base_url=f"http://127.0.0.1:{upstream.server_port}/v1",
            task_jsonl_path=_task_jsonl(tmp_path),
            extra_config={"temperature": 0},
        )

        result = run_legacy_native_ab_sample(
            manifest_path=suite.manifest_path,
            task_ids=("HumanEval_0",),
            repeat=1,
            results_dir=tmp_path / "Results",
            sample_id="native-smoke-sample",
            summary_path=tmp_path / "legacy" / "reports" / "native_summary.json",
            collection_summary_path=tmp_path / "legacy" / "reports" / "native_collection.json",
            timeout_seconds=60,
        )
    finally:
        upstream.shutdown()
        upstream.server_close()

    summary = result.summary
    assert summary["selected_task_ids"] == ("HumanEval_0",)
    assert summary["collection"]["ready_for_ab_report"] is True
    assert summary["collection"]["real_provider_metrics_available"] is True
    assert summary["collection"]["task_metrics_available"] is True
    assert len(upstream_requests) >= 3
    for variant in ("baseline", "plugin_rule_only", "plugin_nl_segmentation"):
        assert summary["variants"][variant]["success_rate"] == 1.0
        assert summary["variants"][variant]["success_evaluated_count"] == 1
        tabulate = summary["variants"][variant]["tabulate_csv_path"]
        assert "HumanEval_0,True" in open(tabulate, encoding="utf-8").read()

    manifest = suite.manifest
    baseline_tasks = _read_jsonl(manifest["oai_config_lists"]["baseline"]["task_results_path"])
    assert baseline_tasks[0]["task_id"] == "HumanEval_0::trial_0"
    rule_summary = json.loads(open(manifest["recommended_ab_eval"]["rule_only_summary"], encoding="utf-8").read())
    assert rule_summary["task_metrics_available"] is True
    assert rule_summary["real_provider_metrics_available"] is True
    assert rule_summary["baseline"]["task_results"]["success_rate"] == 1.0


def test_legacy_native_run_cli_writes_summary(tmp_path) -> None:
    upstream = _start_fake_upstream([])
    try:
        suite = generate_legacy_agbench_suite(
            output_dir=tmp_path / "legacy",
            suite_id="native-cli",
            model="fake-model",
            upstream_base_url=f"http://127.0.0.1:{upstream.server_port}/v1",
            task_jsonl_path=_task_jsonl(tmp_path),
        )
        summary_path = tmp_path / "legacy" / "reports" / "native_cli_summary.json"

        exit_code = main(
            [
                "--manifest",
                suite.manifest_path,
                "--task-id",
                "HumanEval_0",
                "--repeat",
                "1",
                "--results-dir",
                str(tmp_path / "Results"),
                "--sample-id",
                "native-cli-sample",
                "--summary",
                str(summary_path),
                "--timeout-seconds",
                "60",
            ]
        )
    finally:
        upstream.shutdown()
        upstream.server_close()

    assert exit_code == 0
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["schema_version"] == "prefix-legacy-agbench-native-run-v1"
    assert written["selected_task_count"] == 1


def _start_fake_upstream(seen_requests: list[dict[str, Any]]) -> ThreadingHTTPServer:
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
                                    "role": "assistant",
                                    "content": (
                                        "```python\n"
                                        "from my_tests import run_tests\n"
                                        "def identity(x):\n"
                                        "    return x\n"
                                        "run_tests(identity)\n"
                                        "```\nTERMINATE"
                                    ),
                                }
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "total_tokens": 120,
                            "prompt_tokens_details": {"cached_tokens": 64},
                        },
                    }
                ).encode("utf-8")
            )

        def log_message(self, format, *args) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _task_jsonl(tmp_path):
    tasks_dir = tmp_path / "Tasks"
    template_dir = tmp_path / "Templates" / "TwoAgents"
    coding_dir = template_dir / "coding"
    tasks_dir.mkdir()
    coding_dir.mkdir(parents=True)
    (template_dir / "prompt.txt").write_text("__PROMPT__", encoding="utf-8")
    (coding_dir / "my_tests.py").write_text(
        "# ruff: noqa: F821\n"
        "__TEST__\n\n"
        "def run_tests(candidate):\n"
        "    check(candidate)\n"
        "    print('ALL TESTS PASSED !#!#\\nTERMINATE')\n",
        encoding="utf-8",
    )
    (template_dir / "scenario.py").write_text(
        "import autogen\n"
        "import testbed_utils\n"
        "testbed_utils.init()\n"
        "config_list = autogen.config_list_from_json('OAI_CONFIG_LIST')\n"
        "assistant = autogen.AssistantAgent('assistant', llm_config=testbed_utils.default_llm_config(config_list, timeout=30))\n"
        "user_proxy = autogen.UserProxyAgent('user_proxy', human_input_mode='NEVER', code_execution_config={'work_dir': 'coding', 'use_docker': False}, max_consecutive_auto_reply=2, default_auto_reply='TERMINATE')\n"
        "user_proxy.initiate_chat(assistant, message='Return a python code block that imports run_tests and calls it on identity.')\n"
        "testbed_utils.finalize(agents=[assistant, user_proxy])\n",
        encoding="utf-8",
    )
    path = tasks_dir / "human_eval_two_agents.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "HumanEval_0",
                "template": str(template_dir),
                "substitutions": {
                    "prompt.txt": {"__PROMPT__": "def identity(x): return x\n"},
                    "coding/my_tests.py": {
                        "__TEST__": "def check(candidate):\n    assert candidate(3) == 3\n"
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
