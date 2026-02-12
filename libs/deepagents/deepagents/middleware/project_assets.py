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

The following files and resources are available in this project:

{assets_list}

### When to delegate to a subagent with assets
- The user's question requires reading, analyzing, or cross-referencing content from specific files
- The task involves extracting information that only exists inside the assets, not in the conversation history

### When NOT to delegate
- You can already answer from information in the conversation (e.g., a previous subagent already returned the relevant data)
- The user is asking about the asset catalog itself (e.g., "What files do you have?" or "Which assets are CSVs?") — answer directly from this list
- The task is simple enough to handle without file access

### Asset selection strategy
- Match the user's query and full conversation history against asset descriptions to identify relevant files
- Pass only the assets the subagent actually needs — do not send the entire catalog
- Different subagent types may need different asset subsets — route assets to the appropriate subagent
- In multi-turn conversations, carry forward context from earlier messages when selecting assets (e.g., if the user asked about "Sarah" earlier and now says "also check the travel records", include travel-related assets for the same person without re-querying assets that already returned results)
- When a task requires comparing or cross-referencing data within the same subagent's domain, send multiple assets to a single subagent call so it can access all of them together

<examples>
<example>
User: "Tell me about Sarah O'Brien"
Available subagents: pdf-subagent (handles PDFs), csv-subagent (handles CSVs)
Available assets: bank.pdf (bank overview), commerce.pdf (commerce data), employees.csv (employee directory), travel.pdf (travel records)
Reasoning: Sarah likely appears in employee records (employees.csv) and may be referenced in bank documents (bank.pdf). Commerce and travel data are unlikely to mention her by name without further context. Route PDF assets to the pdf-subagent and CSV assets to the csv-subagent.
Action: In parallel — task(subagent_type="pdf-subagent", asset_name=["bank.pdf"], description="Find all information about Sarah O'Brien") AND task(subagent_type="csv-subagent", asset_name=["employees.csv"], description="Look up Sarah O'Brien in the employee directory")
</example>

<example>
User (follow-up after receiving Sarah's info): "Now also check if she has any travel records"
Reasoning: Previous messages already contain Sarah's employee and bank data. Only the travel asset is new. No need to re-query employees.csv or bank.pdf.
Action: task(subagent_type="pdf-subagent", asset_name=["travel.pdf"], description="Find any travel records for Sarah O'Brien")
</example>

<example>
User: "What files are available in this project?"
Reasoning: This is a question about the asset catalog itself, not about content within the files. Answer directly — no delegation needed.
Action: Respond directly listing all available assets and their descriptions.
</example>

<example>
User: "Compare the bank overview with the commerce summary"
Reasoning: Both bank.pdf and commerce.pdf are PDFs that belong to the same subagent. Send both to a single call so the subagent can read and compare them side by side.
Action: task(subagent_type="pdf-subagent", asset_name=["bank.pdf", "commerce.pdf"], description="Compare the key themes and findings between the bank overview and the commerce summary")
</example>
</examples>"""


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
