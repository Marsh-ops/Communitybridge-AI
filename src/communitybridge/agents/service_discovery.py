"""The Service Discovery Agent: finds the nearest service for the confirmed goal.

The project design: this agent "finds the nearest eligible service", may use
`tools/service_query.py` and nothing else, and "every result carries its `source` and
`last_verified`". The project plan lists it as the agent that "finds the nearest eligible service
(resident, visiting or telehealth)" from a read-only curated dataset. It runs once the user has
confirmed the extracted details at Gate 1.

Design points worth being able to explain:

- **It makes no model call, which is a deviation from the design diagram** (that figure shows a model
  here, for "query + summary"). The goal is already a validated request, so there is no query left
  to write, and the result is a database row, so a model's summary could only restate it with a
  chance of getting a distance, a mode or a date wrong. The summary the user sees is built from
  the record itself, so it is always faithful and always carries its source. The role is still a
  distinct responsibility with its own tool, its own verification (source and date on every
  result) and its own trace events; the project brief asks for agent roles "from the
  workflow, not from technology preferences". The way back is to
  add a model-written summary behind a check that it repeats only the record's facts.
- **It reaches its one tool through a `ToolBox`.** The box holds only `service_query`, so a call to
  any other tool is refused and logged (`graph/toolbox.py`). A test also reads this module's
  imports, so it cannot import another tool directly either.
- **"Nearest" is not "eligible".** The tool returns the services that count as nearest under PATS
  §6.4.6 and never says whether the client may use them: only the rules engine decides that. This node can write `service` and the trace, and nothing that looks like a
  decision.
- **Every failure ends the run for staff.** An unknown home town, a missing distance, an unreadable
  database or a specialty nobody offers cannot be fixed by the set re-plans (the next window, or a
  telehealth alternative), because both need a service to exist. So the run ends as `escalated`
  with the reason in the trace. The reason is a fixed sentence, never the exception text, which
  holds the town the user typed: that is personal data and stays out of the trace.
- **A bug is not an escalation.** A missing tool, or a goal that was never confirmed, raises an
  error instead of being passed to staff as if it were a client's problem.
"""

from typing import Any, Literal

from communitybridge.graph.node_contract import NodeFn, traced_node
from communitybridge.graph.state import GraphState
from communitybridge.graph.toolbox import ToolBox
from communitybridge.models import Service, TraceEvent
from communitybridge.models.discovery import ServiceSearch
from communitybridge.tools.distance import DistanceDataError, DistanceNotFoundError
from communitybridge.tools.service_query import ServiceQueryError, UnknownTownError

# The tool's name in the permission map (`AGENT_TOOLS` in graph/build.py) and in the trace.
SERVICE_QUERY = "service_query"


class DiscoveryError(Exception):
    """The graph asked this agent to do something it should never be asked to do (a bug)."""


def _event(task: str, result: str, status: Any) -> TraceEvent:
    return TraceEvent(
        task=task,
        agent="service_discovery",
        tool_or_source=SERVICE_QUERY,
        result=result,
        status=status,
    )


def _escalate(reason: str) -> dict[str, Any]:
    """End the run for staff. Two events: what failed, and the hand-over."""
    return {
        "status": "escalated",
        "trace": [
            _event("find nearest service", reason, "failed"),
            TraceEvent(
                task="pass to staff",
                agent="service_discovery",
                tool_or_source="escalation to staff",
                result="no service could be found; a staff member can take the request",
                status="escalated",
            ),
        ],
    }


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _found(search: ServiceSearch) -> dict[str, Any]:
    chosen = search.nearest[0]  # ties are already ordered by service id
    record = chosen.service
    summary = (
        f"nearest service {record.service_id} ({record.mode}), {chosen.distance_km:g} km by road; "
        f"source: {chosen.citation.source}; last verified {chosen.citation.last_verified}"
    )
    if len(search.nearest) > 1:
        # PATS counts every tied service as the nearest, so eligibility does not change; the
        # others stay in the dataset for Logistics. Said here so the choice is never silent.
        summary += (
            f"; {_plural(len(search.nearest), 'service')} tie for nearest, "
            "the first by service id was used"
        )
    if search.others:
        summary += f"; {_plural(len(search.others), 'farther alternative')} found"
    # A plain Service: the dataset record also holds a visiting specialist's dates, which Logistics
    # reads from the dataset itself and which the graph state has no field for.
    service = Service.model_validate(record.model_dump(exclude={"visit_dates"}))
    return {"service": service, "trace": [_event("find nearest service", summary, "ok")]}


def make_discover_service(tools: ToolBox) -> NodeFn:
    """Build the discover_service node around the tool box the graph gives this agent."""

    @traced_node("service_discovery")
    def discover_service(state: GraphState) -> dict[str, Any]:
        goal = state.goal
        if goal is None:
            raise DiscoveryError("the state has no goal to search for")
        if not goal.confirmed_by_user:
            raise DiscoveryError("the goal has not been confirmed by the user (Gate 1)")
        try:
            search: ServiceSearch = tools.call(SERVICE_QUERY, goal.origin_town, goal.specialty)
        except UnknownTownError:
            return _escalate("the confirmed home town is not in the service dataset")
        except DistanceNotFoundError:
            return _escalate("the dataset has no road distance for this journey")
        except (ServiceQueryError, DistanceDataError):
            return _escalate("the service database could not be read")
        if not search.nearest:
            return _escalate("no service in the dataset offers the confirmed specialty")
        return _found(search)

    return discover_service


def route_after_discovery(state: GraphState) -> Literal["decide", "finish"]:
    """A service found goes on to the eligibility step; an escalated run finishes."""
    if state.status == "escalated":
        return "finish"
    if state.service is None:
        raise DiscoveryError("discovery ended with no service and was not escalated")
    return "decide"
