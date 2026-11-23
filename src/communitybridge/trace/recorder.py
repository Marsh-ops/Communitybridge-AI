"""How trace events join the trace in the graph state. Serves every agent.

The project brief asks for an "Agent trace showing task, source/tool, result and status",
and the project plan makes that trace visible to the user. LangGraph merges what a node returns
into the state with a *reducer*. This one only ever appends, so a node cannot reorder, replace or
delete an earlier event. That is what lets the app keep a consent approval visible after the
consent is withdrawn: the withdrawal is a new event, not an edit.

The trace lives inside the state, so the SQLite checkpoint (graph/checkpoint.py) is also its
permanent record. There is no second trace table to keep in step with it.
"""

from communitybridge.models.trace import TraceEvent


class TraceError(Exception):
    """A trace update was not a list of TraceEvent objects."""


def append_trace(existing: list[TraceEvent], new: list[TraceEvent]) -> list[TraceEvent]:
    """Return `existing` followed by `new`, without changing either list.

    Anything other than a list of TraceEvent is refused. A single event, a dict or a string would
    otherwise be stored and only fail later, when the user interface tried to display the trace.
    """
    if not isinstance(new, list) or not all(isinstance(item, TraceEvent) for item in new):
        raise TraceError("a trace update must be a list of TraceEvent objects")
    return [*existing, *new]
