"""The tool box: how an agent reaches its tools, and only its tools. Serves every agent.

The project design says "Agents call only the tools listed for them. Each tool assignment is
defined in one place (`graph/build.py`)", and the project brief asks to "specify tool permissions".
A rule that only lives in a table can be broken by one wrong import, so this makes it a runtime
fact: an agent is handed a `ToolBox` holding just its own tools, and the only way it can reach a
tool is `box.call(name, ...)`. Asking for any other name raises `ToolPermissionError`.

Design points:
- The permission map (who may call what) is `AGENT_TOOLS` in `graph/build.py`. This module has no
  list of its own, so there is still only one place to read the permissions.
- Every call, allowed or refused, goes in a log. Tests read it to show that an agent called
  exactly the tools it was given, and the evaluation ("tool-permission violations",
  target 0) can count the refused ones.
- A tool's own errors pass straight through. The agent decides how to handle them.
- The box copies the tools it is given, so a dict the caller still holds cannot widen it later.

The box is about honesty and testability, not security against hostile code: an agent module could
still import a tool directly. A second test therefore reads each agent
module's imports, so a direct import of a tool it was not given fails the build.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from communitybridge.models.trace import AgentName

ToolFn = Callable[..., Any]


class ToolPermissionError(Exception):
    """An agent asked for a tool it was not given."""


@dataclass(frozen=True)
class ToolCall:
    """One entry in the log: the tool asked for, and whether the box allowed it."""

    tool: str
    allowed: bool


class ToolBox:
    """The tools one agent may use, and a log of how it used them."""

    def __init__(self, agent: AgentName, tools: Mapping[str, ToolFn]) -> None:
        self.agent = agent
        self._tools = dict(tools)
        self._calls: list[ToolCall] = []

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    @property
    def calls(self) -> list[ToolCall]:
        """Every call so far, in order. A copy: editing it does not change the record."""
        return list(self._calls)

    @property
    def violations(self) -> list[ToolCall]:
        """The calls the box refused."""
        return [call for call in self._calls if not call.allowed]

    def call(self, name: str, /, *args: Any, **kwargs: Any) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            self._calls.append(ToolCall(name, allowed=False))
            allowed = ", ".join(self.tool_names) or "no tools"
            raise ToolPermissionError(
                f"the {self.agent} agent may not call {name!r}; it may call: {allowed}"
            )
        # Logged before the call, so a tool that fails still shows up in the record.
        self._calls.append(ToolCall(name, allowed=True))
        return tool(*args, **kwargs)
