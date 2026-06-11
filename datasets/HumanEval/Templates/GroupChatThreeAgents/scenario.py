import os

import autogen
import testbed_utils


testbed_utils.init()

work_dir = "coding"

with open("prompt.txt", "rt", encoding="utf-8") as fh:
    PROMPT = fh.read()

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
llm_config = testbed_utils.default_llm_config(config_list, timeout=240)

shared_context = """
SHARED_GROUPCHAT_CONTEXT_START
The team is solving one HumanEval programming task inside AutoGenBench. All worker agents receive the same task prompt, the same hidden unit-test goal, the same collaboration rules, the same evidence standard, the same output schema, and the same risk boundary. The planner should summarize the implementation intent, the engineer should refine edge cases, and the coder should output exactly one executable Python code block. The final code must import run_tests from my_tests, define the required entry point, and call run_tests on that entry point. Do not invent test results. Do not ask for external credentials. Keep the response deterministic.
SHARED_GROUPCHAT_CONTEXT_END
SHARED_GROUPCHAT_CONTEXT_START
The team is solving one HumanEval programming task inside AutoGenBench. All worker agents receive the same task prompt, the same hidden unit-test goal, the same collaboration rules, the same evidence standard, the same output schema, and the same risk boundary. The planner should summarize the implementation intent, the engineer should refine edge cases, and the coder should output exactly one executable Python code block. The final code must import run_tests from my_tests, define the required entry point, and call run_tests on that entry point. Do not invent test results. Do not ask for external credentials. Keep the response deterministic.
SHARED_GROUPCHAT_CONTEXT_END
SHARED_GROUPCHAT_CONTEXT_START
The team is solving one HumanEval programming task inside AutoGenBench. All worker agents receive the same task prompt, the same hidden unit-test goal, the same collaboration rules, the same evidence standard, the same output schema, and the same risk boundary. The planner should summarize the implementation intent, the engineer should refine edge cases, and the coder should output exactly one executable Python code block. The final code must import run_tests from my_tests, define the required entry point, and call run_tests on that entry point. Do not invent test results. Do not ask for external credentials. Keep the response deterministic.
SHARED_GROUPCHAT_CONTEXT_END
TOOL_SCHEMA_START
Available local benchmark actions are represented as static context only: read_prompt(path), infer_solution(prompt), write_python_code(entry_point), execute_unit_tests(work_dir), and report_termination_marker(). These tools are not executed by the model; this schema is repeated so shared context can be observed in multi-agent request prefixes.
TOOL_SCHEMA_END
OUTPUT_FORMAT_START
Planner and engineer: write one concise paragraph only. Coder: output one stand-alone Python code block and then TERMINATE. The code block must be directly executable in Python.
OUTPUT_FORMAT_END
"""


def agent_system(agent_name, role_note):
    return f"""ROLE_SPECIFIC_INSTRUCTION_START
AGENT_NAME: {agent_name}
{role_note}
ROLE_SPECIFIC_INSTRUCTION_END

USER_TASK_START
Complete the HumanEval function below so the AutoGenBench unit tests pass.
{PROMPT}
USER_TASK_END

{shared_context}

CURRENT_TURN_INSTRUCTION_START
Respond with the contribution required by your role. Preserve the task semantics exactly.
CURRENT_TURN_INSTRUCTION_END"""


planner = autogen.AssistantAgent(
    "planner",
    system_message=agent_system(
        "planner",
        "You are the Planner Agent. State the algorithm and the exact edge cases the implementation must cover. Do not output code.",
    ),
    llm_config=llm_config,
)

engineer = autogen.AssistantAgent(
    "engineer",
    system_message=agent_system(
        "engineer",
        "You are the Engineer Agent. Refine the implementation details and mention pitfalls. Do not output code.",
    ),
    llm_config=llm_config,
)

coder = autogen.AssistantAgent(
    "coder",
    system_message=agent_system(
        "coder",
        "You are the Coder Agent. Output exactly one complete Python code block that imports run_tests, defines __ENTRY_POINT__, calls run_tests(__ENTRY_POINT__), and then write TERMINATE.",
    ),
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: str(x.get("content", "")).find("TERMINATE") >= 0,
    code_execution_config={
        "work_dir": work_dir,
        "use_docker": False,
        "last_n_messages": 1,
    },
    max_consecutive_auto_reply=2,
    default_auto_reply="TERMINATE",
)

initial_message = """
The following Python code imports run_tests(candidate) from my_tests.py and runs it on __ENTRY_POINT__.
Complete __ENTRY_POINT__ and make the benchmark tests pass. The final coder response must be one stand-alone
Python code block that can be run directly.

```python
from my_tests import run_tests

""" + PROMPT + """

run_tests(__ENTRY_POINT__)
```
"""

groupchat = autogen.GroupChat(
    agents=[planner, engineer, coder, user_proxy],
    messages=[],
    max_round=5,
    speaker_selection_method="round_robin",
    allow_repeat_speaker=False,
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=False)

user_proxy.initiate_chat(manager, message=initial_message)

testbed_utils.finalize(agents=[planner, engineer, coder, user_proxy, manager])
