"""Tests for ProjectAssetsMiddleware functionality."""

from collections.abc import Callable
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from deepagents.graph import create_deep_agent
from deepagents.middleware.project_assets import (
    ProjectAsset,
    ProjectAssetsMiddleware,
    ProjectAssetsState,
)
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, _EXCLUDED_STATE_KEYS
from tests.unit_tests.chat_model import GenericFakeChatModel
from tests.utils import assert_all_deepagent_qualities


SAMPLE_ASSETS: list[ProjectAsset] = [
    {"name": "bank.pdf", "description": "Overview of a fictional bank"},
    {"name": "commerce.pdf", "description": "Summary of fictional commerce data"},
    {"name": "employees.csv", "description": "Sample employee directory"},
]


class SystemMessageCapturingMiddleware(AgentMiddleware):
    """Middleware that captures the system message for testing purposes."""

    def __init__(self) -> None:
        self.captured_system_messages: list = []

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        if request.system_message is not None:
            self.captured_system_messages.append(request.system_message)
        return handler(request)


class TestProjectAssetsMiddleware:
    """Tests for the ProjectAssetsMiddleware."""

    def test_project_assets_in_excluded_state_keys(self) -> None:
        """Verify that 'project_assets' is in _EXCLUDED_STATE_KEYS.

        This prevents project_assets from leaking from parent to subagent
        or from subagent output back to parent.
        """
        assert "project_assets" in _EXCLUDED_STATE_KEYS

    def test_middleware_injects_assets_into_system_prompt(self) -> None:
        """Test that wrap_model_call injects asset catalog into the system prompt.

        Verifies the end-to-end flow:
        1. ProjectAssetsMiddleware is configured with sample assets
        2. On each model call, asset descriptions are injected into system prompt
        3. The LLM sees all asset names and descriptions
        """
        capture_mw = SystemMessageCapturingMiddleware()

        chat_model = GenericFakeChatModel(
            messages=iter([AIMessage(content="I can see the project assets.")])
        )

        agent = create_deep_agent(
            model=chat_model,
            project_assets=SAMPLE_ASSETS,
            middleware=[capture_mw],
            checkpointer=InMemorySaver(),
        )

        agent.invoke(
            {"messages": [HumanMessage(content="What files do you have?")]},
            config={"configurable": {"thread_id": "test_inject"}},
        )

        assert len(capture_mw.captured_system_messages) > 0, "System message should have been captured"
        system_text = str(capture_mw.captured_system_messages[0].content)
        assert "bank.pdf" in system_text
        assert "commerce.pdf" in system_text
        assert "employees.csv" in system_text
        assert "Project Assets" in system_text

    def test_create_deep_agent_with_project_assets(self) -> None:
        """Test that create_deep_agent accepts project_assets parameter and compiles."""
        agent = create_deep_agent(project_assets=SAMPLE_ASSETS)
        assert_all_deepagent_qualities(agent)

    def test_create_deep_agent_without_project_assets(self) -> None:
        """Test backward compatibility: create_deep_agent works without project_assets."""
        agent = create_deep_agent()
        assert_all_deepagent_qualities(agent)

    def test_standalone_middleware_usage(self) -> None:
        """Test that ProjectAssetsMiddleware can be used standalone via middleware param."""
        agent = create_deep_agent(
            middleware=[ProjectAssetsMiddleware(assets=SAMPLE_ASSETS)],
        )
        assert_all_deepagent_qualities(agent)

    def test_asset_names_passed_to_subagent_via_task_tool(self) -> None:
        """End-to-end test: LLM calls task tool with asset_name, subagent receives them.

        Flow:
        1. Deep agent has project_assets configured
        2. LLM decides to call task tool with asset_name=["bank.pdf", "employees.csv"]
        3. Subagent receives asset_name in its runtime.state
        """
        captured_subagent_states: list[dict[str, Any]] = []

        @tool
        def capture_state(query: str, runtime: ToolRuntime) -> str:
            """Captures runtime state from the subagent."""
            captured_subagent_states.append(dict(runtime.state))
            return f"Processed: {query}"

        subagent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "capture_state",
                                "args": {"query": "check assets"},
                                "id": "call_cap",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Subagent done with assets."),
                ]
            )
        )

        leader_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Look up Sarah in bank and employee data",
                                    "subagent_type": "data-agent",
                                    "asset_name": ["bank.pdf", "employees.csv"],
                                },
                                "id": "call_task",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Here is Sarah's info."),
                ]
            )
        )

        leader = create_deep_agent(
            model=leader_model,
            project_assets=SAMPLE_ASSETS,
            checkpointer=InMemorySaver(),
            subagents=[
                SubAgent(
                    name="data-agent",
                    description="Agent that processes data files",
                    system_prompt="You process data files.",
                    model=subagent_model,
                    tools=[capture_state],
                )
            ],
        )

        leader.invoke(
            {"messages": [HumanMessage(content="Tell me about Sarah")]},
            config={"configurable": {"thread_id": "test_e2e_assets"}},
        )

        assert len(captured_subagent_states) > 0, "Subagent tool should have been invoked"
        subagent_state = captured_subagent_states[0]
        assert "asset_name" in subagent_state
        assert subagent_state["asset_name"] == ["bank.pdf", "employees.csv"]

    def test_project_assets_not_in_subagent_state(self) -> None:
        """Test that project_assets from parent does not leak into subagent state.

        project_assets is in _EXCLUDED_STATE_KEYS, so even if the parent agent
        has it in state, it should not appear in the subagent's runtime.state.
        """
        captured_subagent_states: list[dict[str, Any]] = []

        @tool
        def capture_state(query: str, runtime: ToolRuntime) -> str:
            """Captures runtime state."""
            captured_subagent_states.append(dict(runtime.state))
            return "Done"

        subagent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "capture_state",
                                "args": {"query": "check"},
                                "id": "call_cap",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        leader_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Do work",
                                    "subagent_type": "worker",
                                },
                                "id": "call_task",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        leader = create_deep_agent(
            model=leader_model,
            project_assets=SAMPLE_ASSETS,
            checkpointer=InMemorySaver(),
            subagents=[
                SubAgent(
                    name="worker",
                    description="Worker agent",
                    system_prompt="You do work.",
                    model=subagent_model,
                    tools=[capture_state],
                )
            ],
        )

        leader.invoke(
            {"messages": [HumanMessage(content="Go")]},
            config={"configurable": {"thread_id": "test_no_leak"}},
        )

        assert len(captured_subagent_states) > 0
        subagent_state = captured_subagent_states[0]
        assert "project_assets" not in subagent_state, (
            "project_assets should NOT leak from parent to subagent state"
        )
