from __future__ import annotations

import json

from autogen_prefix_tree.source_prompt_corpus import collect_source_prompts, evaluate_source_prompt_corpus


def test_source_prompt_corpus_extracts_prompts_and_runs_dataset_eval(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    module_path = source_root / "agents.py"
    module_path.write_text(
        '''
DEFAULT_SYSTEM_MESSAGE = """
You are a task verification assistant.
Check progress carefully and respond with TERMINATE when done.
"""

def build_agent():
    return AssistantAgent(
        name="researcher",
        system_message="""You are a research assistant.
Use tools carefully.
Always verify information across multiple sources.
""",
    )

selector_prompt = """You are coordinating a research team.
The following roles are available:
{roles}
Read the following conversation:
{history}
Only return the role.
"""
''',
        encoding="utf-8",
    )
    test_dir = source_root / "tests"
    test_dir.mkdir()
    (test_dir / "test_prompt.py").write_text(
        'SYSTEM_MESSAGE = "You are a test assistant. TERMINATE when done."\n',
        encoding="utf-8",
    )

    records = tuple(collect_source_prompts(source_root))
    result = evaluate_source_prompt_corpus(
        source_root=source_root,
        output_requests_path=tmp_path / "requests.jsonl",
        summary_path=tmp_path / "summary.json",
        telemetry_path=tmp_path / "telemetry.jsonl",
        session_id="source-corpus-test",
    )

    assert len(records) == 3
    assert result.prompt_count == 3
    assert result.source_file_count == 1
    assert result.summary["supported_request_count"] == 3
    assert result.summary["semantic_coverage_supported"] is True
    assert result.summary["rule_gap_diagnostics"]["repeated_nonprefix_system_block_count"] == 0
    assert result.summary["source_prompt_corpus"]["prompt_count"] == 3
    assert result.summary["source_prompt_corpus"]["source_file_count"] == 1
    assert result.summary["source_prompt_corpus"]["prompt_sources"][0]["content_hash"]
    request_text = (tmp_path / "requests.jsonl").read_text(encoding="utf-8")
    assert "test assistant" not in request_text
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["input_record_count"] == 3
    assert summary["source_prompt_corpus"]["prompt_count"] == 3


def test_source_prompt_corpus_exports_prompt_safe_semantic_candidates_by_default(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    module_path = source_root / "agent.py"
    module_path.write_text(
        '''
SYSTEM_MESSAGE = """
You are a helpful AI assistant.
When using code, you must indicate the script type in the code block.
Do not ask users to copy and paste the result.
When you find an answer, verify the answer carefully and include evidence.
"""
''',
        encoding="utf-8",
    )
    candidates_path = tmp_path / "candidates.jsonl"

    evaluate_source_prompt_corpus(
        source_root=source_root,
        output_requests_path=tmp_path / "requests.jsonl",
        summary_path=tmp_path / "summary.json",
        telemetry_path=tmp_path / "telemetry.jsonl",
        candidates_path=candidates_path,
        session_id="candidate-export",
    )

    rows = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert rows[0]["schema_version"] == "prefix-semantic-candidate-v1"
    assert "text" not in rows[0]
    assert rows[0]["label"] is None
    assert "copy and paste" not in json.dumps(rows, ensure_ascii=False)


def test_source_prompt_corpus_can_explicitly_export_candidate_text(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    module_path = source_root / "agent.py"
    module_path.write_text(
        '''
SYSTEM_MESSAGE = """
You are a helpful AI assistant.
When using code, you must indicate the script type in the code block.
"""
''',
        encoding="utf-8",
    )
    candidates_path = tmp_path / "candidates_with_text.jsonl"

    evaluate_source_prompt_corpus(
        source_root=source_root,
        output_requests_path=tmp_path / "requests.jsonl",
        summary_path=tmp_path / "summary.json",
        telemetry_path=tmp_path / "telemetry.jsonl",
        candidates_path=candidates_path,
        include_candidate_text=True,
        session_id="candidate-export-text",
    )

    rows = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert any("code block" in row.get("text", "") for row in rows)


def test_source_prompt_corpus_extracts_short_strong_instruction_defaults(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "types.py").write_text(
        '''
class Agent:
    instructions: str = "You are a helpful agent."

class RealtimeAgent:
    def __init__(self, sys_prompt: str = "You are a helpful assistant."):
        self.sys_prompt = sys_prompt
''',
        encoding="utf-8",
    )

    records = tuple(collect_source_prompts(source_root))

    assert {record.symbol for record in records} == {
        "Agent.instructions",
        "RealtimeAgent.__init__.sys_prompt",
    }
    assert all(record.content.startswith("You are") for record in records)


def test_source_prompt_corpus_extracts_json_prompt_resources(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "en.json").write_text(
        json.dumps(
            {
                "slices": {
                    "observation": "\nObservation:",
                    "role_playing": "You are {role}. {backstory}\nYour personal goal is: {goal}",
                    "tools": (
                        "You ONLY have access to the following tools:\n{tools}\n\n"
                        "Use this format:\nThought: reason\nAction: one tool\nAction Input: JSON"
                    ),
                },
                "ui": {
                    "api_key_label": "Enter API key",
                    "button": "Submit",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    records = tuple(collect_source_prompts(source_root))

    assert {record.symbol for record in records} == {"slices.role_playing", "slices.tools"}
    assert all(record.extraction == "json_prompt_value" for record in records)
    serialized = json.dumps([record.__dict__ for record in records], ensure_ascii=False)
    assert "Enter API key" not in serialized
    assert "Observation:" not in serialized


def test_source_prompt_corpus_json_extraction_keeps_prompt_paths_only(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "gallery.json").write_text(
        json.dumps(
            {
                "components": {
                    "agents": [
                        {
                            "description": "An agent that provides assistance with ability to use tools.",
                            "config": {
                                "system_message": "You are a helpful assistant. Solve tasks carefully.",
                                "source_code": (
                                    "def tool():\n"
                                    "    return 'Assistant can use this helper when asked.'\n"
                                ),
                            },
                            "tools": [
                                {
                                    "config": {
                                        "source_code": (
                                            "def helper():\n"
                                            "    return 'You must use this helper carefully.'\n"
                                        )
                                    }
                                }
                            ],
                        }
                    ]
                },
                "properties": {
                    "surfaceId": {
                        "description": (
                            "The unique identifier for the UI surface to be updated. "
                            "A client MUST use this identifier."
                        )
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    records = tuple(collect_source_prompts(source_root))

    assert [record.symbol for record in records] == ["components.agents.0.config.system_message"]


def test_source_prompt_corpus_extracts_static_concatenated_and_fstring_prompt_text(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "formatter.py").write_text(
        '''
class Formatter:
    def __init__(
        self,
        conversation_history_prompt: str = (
            "# Conversation History\\n"
            "The content between <history></history> tags contains "
            "your conversation history\\n"
        ),
    ):
        self.conversation_history_prompt = conversation_history_prompt

def build_hint(block):
    return TextBlock(
        text=f"<system-info>The following are the image contents from the tool result of '{block['name']}':",
    )
''',
        encoding="utf-8",
    )

    records = tuple(collect_source_prompts(source_root))

    assert any(record.symbol == "Formatter.__init__.conversation_history_prompt" for record in records)
    assert any(record.symbol == "build_hint.text" for record in records)
    assert any("{block['name']}" in record.content for record in records)
