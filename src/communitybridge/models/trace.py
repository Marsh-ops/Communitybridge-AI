"""TraceEvent: one row of the visible agent trace.

The project brief asks for an "Agent trace showing task, source/tool, result and status",
and the project plan makes the trace visible to the user. Agent names and statuses are fixed
lists, so a misspelt or unknown agent can never appear in the trace.
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from communitybridge.models.common import NonEmptyStr, StrictModel

AgentName = Literal[
    "coordinator", "service_discovery", "eligibility", "logistics", "verifier", "consent"
]
# "withdrawn" records a consent withdrawal so the user sees it in the trace too.
TraceStatus = Literal[
    "ok", "failed", "blocked", "awaiting_approval", "approved", "withdrawn", "escalated"
]


def _now_utc() -> datetime:
    return datetime.now(UTC)


class TraceEvent(StrictModel):
    """One visible step of a run: the task, which agent did it, the tool or source, and the result.

    Every graph node writes one and the trace page shows them, so auditing is
    a feature the person can see, not only a log.
    """

    task: NonEmptyStr
    agent: AgentName
    tool_or_source: NonEmptyStr
    result: NonEmptyStr
    status: TraceStatus
    # Timezone-aware so events from different steps sort correctly.
    timestamp: datetime = Field(default_factory=_now_utc)
