from __future__ import annotations

import asyncio
import json

from autogen_prefix_tree import load_jsonl_telemetry
from autogen_prefix_tree.microbench import run_microbenchmark


def test_microbenchmark_writes_telemetry_and_summary(tmp_path) -> None:
    telemetry_path = tmp_path / "requests.jsonl"
    summary_path = tmp_path / "summary.json"

    result = asyncio.run(
        run_microbenchmark(
            telemetry_path=telemetry_path,
            summary_path=summary_path,
            agents=("planner", "engineer", "reviewer"),
            repeats=1,
            session_id="unit-microbench",
        )
    )

    records = load_jsonl_telemetry(telemetry_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert result.telemetry_path == str(telemetry_path)
    assert len(records) == 3
    assert summary["request_count"] == 3
    assert summary["agent_count"] == 3
    assert summary["repeats"] == 1
    assert summary["session_id"] == "unit-microbench"
    assert summary["validation_reason_counts"] == {"no_rewrite_needed": 1, "validated": 2}
    assert summary["applied_count"] == 2
    assert summary["fallback_count"] == 0
    assert summary["total_estimated_gain_chars"] > 0
    assert records[0]["validation"]["reason"] == "no_rewrite_needed"
    assert records[1]["validation"]["reason"] == "validated"
