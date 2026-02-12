"""Middleware for injecting project asset metadata into the agent's system prompt."""

from collections.abc import Awaitable, Callable
from typing import Annotated, NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from deepagents.middleware._utils import append_to_system_message

PROJECT_ASSETS_SYSTEM_PROMPT = """## Project Assets

The following files and resources are available in this project. When delegating tasks to subagents using the `task` tool, select only the assets relevant to the current query and pass their names via the `asset_name` parameter.

{assets_list}

When choosing assets for a subagent task:
- Review the user's current question and the conversation history
- Select only the assets whose descriptions indicate relevance to the task
- Pass the selected asset names as `asset_name` in the task tool call
- Different subagents may need different subsets of assets"""


class ProjectAsset(TypedDict):
    """A project asset descriptor with a name and description."""

    name: str
    """File or resource name (e.g., ``"bank.pdf"``, ``"employees.csv"``)."""

    description: str
    """Brief description of the asset's contents."""


class ProjectAssetsState(AgentState):
    """State for the project assets middleware."""

    project_assets: NotRequired[Annotated[list[ProjectAsset], PrivateStateAttr]]
    """List of project asset descriptors. Not propagated to parent agents."""


class _ProjectAssetsStateUpdate(TypedDict):
    """State update returned by ``before_agent``."""

    project_assets: list[ProjectAsset]


class ProjectAssetsMiddleware(AgentMiddleware):
    """Middleware that injects a project asset catalog into the agent's system prompt.

    This middleware makes the agent aware of all project files and resources so it
    can intelligently select relevant assets when delegating tasks to subagents via
    the ``task`` tool's ``asset_name`` parameter.

    The middleware:

    1. Populates ``project_assets`` in agent state via ``before_agent``
    2. Injects a formatted asset catalog into the system prompt via ``wrap_model_call``
    3. The LLM then selects relevant assets per subagent call based on conversation context

    Example:
        ```python
        from deepagents import create_deep_agent
        from deepagents.middleware.project_assets import ProjectAssetsMiddleware

        agent = create_deep_agent(
            middleware=[
                ProjectAssetsMiddleware(assets=[
                    {"name": "bank.pdf", "description": "Overview of bank services"},
                    {"name": "employees.csv", "description": "Employee directory"},
                ]),
            ],
            subagents=[...],
        )
        ```

    Args:
        assets: List of project asset descriptors. Each must have ``name`` and
            ``description`` keys.
    """

    state_schema = ProjectAssetsState

    def __init__(self, *, assets: list[ProjectAsset]) -> None:
        """Initialize the project assets middleware.

        Args:
            assets: List of project asset descriptors.
        """
        super().__init__()
        self._assets = assets

    @staticmethod
    def _format_assets_list(assets: list[ProjectAsset]) -> str:
        """Format assets into a readable list for the system prompt."""
        return "\n".join(
            f"- name: {asset['name']}, description: {asset['description']}" for asset in assets
        )

    def before_agent(
        self,
        state: ProjectAssetsState,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> _ProjectAssetsStateUpdate | None:
        """Populate project assets in agent state.

        Only runs on the first call; skips if ``project_assets`` is already present.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with ``project_assets`` populated, or ``None`` if already present.
        """
        if "project_assets" in state:
            return None
        return _ProjectAssetsStateUpdate(project_assets=self._assets)

    async def abefore_agent(
        self,
        state: ProjectAssetsState,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> _ProjectAssetsStateUpdate | None:
        """Populate project assets in agent state (async).

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with ``project_assets`` populated, or ``None`` if already present.
        """
        if "project_assets" in state:
            return None
        return _ProjectAssetsStateUpdate(project_assets=self._assets)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject the project asset catalog into the system prompt.

        Args:
            request: Model request being processed.
            handler: Handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        assets: list[ProjectAsset] = request.state.get("project_assets", [])
        if assets:
            assets_list = self._format_assets_list(assets)
            section = PROJECT_ASSETS_SYSTEM_PROMPT.format(assets_list=assets_list)
            new_system_message = append_to_system_message(request.system_message, section)
            return handler(request.override(system_message=new_system_message))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Inject the project asset catalog into the system prompt (async).

        Args:
            request: Model request being processed.
            handler: Async handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        assets: list[ProjectAsset] = request.state.get("project_assets", [])
        if assets:
            assets_list = self._format_assets_list(assets)
            section = PROJECT_ASSETS_SYSTEM_PROMPT.format(assets_list=assets_list)
            new_system_message = append_to_system_message(request.system_message, section)
            return await handler(request.override(system_message=new_system_message))
        return await handler(request)
