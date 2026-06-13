from __future__ import annotations

from types import SimpleNamespace

from autogen_core.models import SystemMessage

from autogen_prefix_tree import performance_eval_dspro as perf


def test_deepseek_prompt_cache_usage_is_parsed() -> None:
    usage = perf._usage_from_response(
        {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 25,
                "total_tokens": 1025,
                "prompt_cache_hit_tokens": 256,
                "prompt_cache_miss_tokens": 744,
            }
        }
    )

    assert usage["input_tokens"] == 1000
    assert usage["output_tokens"] == 25
    assert usage["cached_tokens"] == 256
    assert usage["cache_miss_tokens"] == 744
    assert usage["cached_tokens_present"] is True


def test_planner_only_default_does_not_force_front_loaded_prefix(monkeypatch) -> None:
    _patch_minimal_planner(monkeypatch)

    def fail_if_called(*_: object, **__: object) -> str:
        raise AssertionError("front-load override must be opt-in")

    monkeypatch.setattr(perf, "_front_load_common_prefix", fail_if_called)

    chosen = perf._choose_optimized_prompts(
        prepared=[_prepared_item()],
        utility_model=None,
        planner_ranker=None,
    )

    assert chosen["task_1"]["prompt"].startswith("TASK_FIRST")


def test_force_front_loaded_prefix_is_explicit_opt_in(monkeypatch) -> None:
    _patch_minimal_planner(monkeypatch)
    calls: list[tuple[str, str]] = []

    def front_load(prompt: str, *, cache_isolation_tag: str = "", task_id: str = "") -> str:
        calls.append((cache_isolation_tag, task_id))
        return "FRONT_LOADED\n" + prompt

    monkeypatch.setattr(perf, "_front_load_common_prefix", front_load)

    chosen = perf._choose_optimized_prompts(
        prepared=[_prepared_item()],
        utility_model=None,
        planner_ranker=None,
        force_front_loaded_common_prefix=True,
    )

    assert calls == [("run-tag", "task_1")]
    assert chosen["task_1"]["prompt"].startswith("FRONT_LOADED")


def _prepared_item() -> dict[str, object]:
    return {
        "task": {
            "task_id": "task_1",
            "dataset_source": "synthetic_agent",
            "scenario_type": "tool_use_permission",
            "source_metadata": {"cache_isolation_tag": "run-tag"},
        },
        "messages": (SystemMessage(content="BASELINE_FIRST\n\n" + perf.GLOBAL_SHARED_BLOCK),),
        "baseline_prompt": "BASELINE_FIRST\n\n" + perf.GLOBAL_SHARED_BLOCK,
    }


def _patch_minimal_planner(monkeypatch) -> None:
    candidate = SimpleNamespace(
        candidate_id="candidate_1",
        generation_reason="balanced:test",
        agent_block_orders={},
        block_text_by_id={},
    )
    plan = SimpleNamespace(estimated_cache_gain=42.0)

    class FakeCompiler:
        def compile(self, messages, session_id):  # noqa: ANN001
            return SimpleNamespace(messages=messages)

    class FakePlanner:
        def _plan_from_candidate(self, compile_result, candidate):  # noqa: ANN001
            return plan

        def observe(self, compile_result, *, session_id):  # noqa: ANN001
            return None

    class FakeValidator:
        def __init__(self, **_: object) -> None:
            pass

        def validate(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(
                applied=True,
                fallback=False,
                reason="ok",
                hard_constraint_passed=True,
                cache_hit_increased=True,
                utility_estimate=None,
                utility_status="passed",
            )

    monkeypatch.setattr(perf, "LocalPromptCompiler", FakeCompiler)
    monkeypatch.setattr(perf, "HierarchicalPrefixPlanner", FakePlanner)
    monkeypatch.setattr(perf, "CacheUtilityValidator", FakeValidator)
    monkeypatch.setattr(perf, "_expanded_candidates_for_task", lambda **_: [candidate])
    monkeypatch.setattr(
        perf,
        "_messages_for_plan",
        lambda **_: (SystemMessage(content="TASK_FIRST\n\n" + perf.GLOBAL_SHARED_BLOCK),),
    )
    monkeypatch.setattr(perf, "runtime_feature_row_from_gate_inputs", lambda **_: {})
    monkeypatch.setattr(
        perf,
        "serialize_prefix_tree_candidate",
        lambda candidate, include_text=False: {"candidate_id": candidate.candidate_id, "include_text": include_text},
    )
