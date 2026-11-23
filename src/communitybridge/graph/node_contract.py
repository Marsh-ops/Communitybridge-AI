"""The contract every graph node must meet before LangGraph saves its result.

The project design says "Every node writes a TraceEvent", and the project brief asks for a
visible agent trace. A probe of LangGraph 1.2.19 showed that it saves whatever a node returns
without checking it against the state model: a node that returned `replan_count: 99` was accepted.
So a node that forgot its trace event, or returned an impossible value, would be saved silently and
only show up later as a gap in the trace the user sees.

`traced_node(agent)` wraps a node function and checks its update *before* LangGraph merges it:
1. it is a dict;
2. it contains at least one TraceEvent, as a proper list;
3. every event names the agent the node belongs to, so the trace can't blame the wrong agent;
4. the state that would result is valid (right fields, right types, the GraphState rules).

A failed check raises `NodeContractError`. LangGraph then stops the run before saving the update,
so the checkpoint still holds the state after the last good node.

Errors raised by the node itself (a tool failure, or the GraphInterrupt that `interrupt()` uses to
pause at a gate) pass straight through: the agent decides how to handle its own tool errors, and
the wrapper only checks what a finished node returns.
"""

import functools
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from communitybridge.graph.state import GraphState
from communitybridge.models.trace import AgentName
from communitybridge.trace.recorder import TraceError, append_trace

NodeFn = Callable[[GraphState], dict[str, Any]]

_MAX_PROBLEMS_SHOWN = 5


class NodeContractError(Exception):
    """A node returned an update that breaks the node contract."""


def traced_node(agent: AgentName) -> Callable[[NodeFn], NodeFn]:
    """Decorator: check a node's update against the contract. `agent` is who the node acts as."""

    def decorate(node: NodeFn) -> NodeFn:
        @functools.wraps(node)
        def checked(state: GraphState) -> dict[str, Any]:
            update = node(state)
            _check(agent, node.__name__, state, update)
            return update

        return checked

    return decorate


def _check(agent: str, name: str, state: GraphState, update: Any) -> None:
    if not isinstance(update, dict):
        raise NodeContractError(
            f"node {name!r} must return a dict of state updates, got {type(update).__name__}"
        )
    if not update.get("trace"):
        raise NodeContractError(
            f"node {name!r} wrote no TraceEvent: every node must record what it did"
        )
    try:
        trace = append_trace(state.trace, update["trace"])
    except TraceError as exc:
        raise NodeContractError(f"node {name!r} wrote a bad trace update: {exc}") from exc

    # Only the new events are checked: the earlier ones were checked when their node ran.
    foreign = sorted({e.agent for e in update["trace"] if e.agent != agent})
    if foreign:
        raise NodeContractError(
            f"node {name!r} acts as the {agent} agent but wrote trace events for "
            f"{', '.join(foreign)}"
        )

    # Validate the state the update would produce. Rebuilding from a dump is the same pattern
    # as ConsentRequest.withdraw(), and it runs the GraphState rules on the merged result.
    merged = {**state.model_dump(), **update, "trace": trace}
    try:
        GraphState.model_validate(merged)
    except ValidationError as exc:
        problems = describe_problems(exc)
        raise NodeContractError(
            f"node {name!r} returned an update that does not fit the state: {problems}"
        ) from exc


def describe_problems(error: ValidationError) -> str:
    """Name each problem as `field: rule`, without echoing the value that broke it.

    A value can be personal data and this message can end up in the trace, so only the field and
    the rule are shown (the same approach as the LLM retry message in llm/structured.py).
    """
    found = error.errors()
    shown = []
    for problem in found[:_MAX_PROBLEMS_SHOWN]:
        field = ".".join(str(part) for part in problem["loc"])
        shown.append(f"{field}: {problem['msg']}" if field else problem["msg"])
    text = "; ".join(shown)
    extra = len(found) - _MAX_PROBLEMS_SHOWN
    return f"{text}; and {extra} more" if extra > 0 else text
