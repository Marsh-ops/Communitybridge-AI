"""The Logistics Agent: plans the journey, asks for approval, then makes SIMULATED requests.

The project design: this agent "matches transport, escort, accommodation and appointment windows",
may use `tools/logistics.py` and the simulated transport, accommodation and calendar APIs, and has
a human approval gate: "Yes, before any (simulated) booking request". The project plan's fourth
in-scope step is to suggest "transport, accommodation and appointment windows that work together".
This module has three parts, in the order the run reaches them:

1. `plan_logistics` asks the planning tool for the travel mode PATS pays for and the windows that
   fit the client's dates, and stores them as options. When the Coordinator has re-planned
   it plans for the attempt the Coordinator chose: the next window's dates, or the telehealth
   fallback inside the client's own dates. It never decides which attempt that is.
2. Gate 3 (`announce_approve_booking`, `approve_booking`) shows the options and waits for the user
   to pick one and approve a simulated booking request, or decline.
3. `request_booking` makes the SIMULATED requests for the chosen option, and only if the trace holds
   the approval. It makes only the kinds of request the user approved at Gate 4 (the
   Consent Agent, `agents/consent.py`), so a declined action is never requested; the Consent Agent
   then releases the data for the requests that were made.

Design points worth being able to explain:

- **It makes no model call, which is a second deviation from the design diagram** (the figure shows a
  model here, "explain options"). Every figure in a journey plan is money or a date from a tool, so
  a model's summary could only restate it with a chance of getting a number wrong, and the user
  decides on those numbers. The way back is the one the Eligibility Agent uses: a model-written
  summary behind a check against the tool's result.
- **No booking without an approval, enforced three times.** The booking node refuses (it raises, it
  does not escalate) unless the trace holds a Gate 3 approval, and hands the simulated services the
  position of that approval as the approval id; the booking tool refuses an empty id before it
  touches anything; and each simulated service refuses it again. `trace/audit.py` can then read any
  finished trace and list booking events with no approval before them: the "zero unapproved
  external actions" figure.
- **The user picks the option.** The state's `chosen_option` is only ever written by Gate 3, from
  the user's answer. Nothing the model or the tools return can choose for them.
- **A booking is all or nothing.** The tool checks every part is still available before it makes
  the first request. If one is not, nothing is booked and the run goes to staff.
- **Failures go to staff with a fixed reason**, never the exception text, which can hold the user's
  town. A bug (an unconfirmed goal, no approval) raises instead of being passed to staff.
  A plan with no option that fits is not an error: it goes to the re-planning step.

The trace never holds the user's words, town or dates of birth: booking events carry the SIMULATED
reference and the approval id, and the full description stays in `state.bookings`.
"""

from typing import Any, Literal, Self

from langgraph.types import interrupt
from pydantic import ValidationError, model_validator

from communitybridge.graph.node_contract import NodeFn, describe_problems, traced_node
from communitybridge.graph.state import GraphState
from communitybridge.graph.toolbox import ToolBox
from communitybridge.models import StrictModel, TraceEvent
from communitybridge.models.consent import ACTION_FOR_KIND, KIND_FOR_ACTION
from communitybridge.models.contact import ClientContact
from communitybridge.models.journey import JourneyPlan
from communitybridge.models.replan import window_for
from communitybridge.models.simulated import BookingRequest
from communitybridge.simulated.common import (
    InvalidRequestError,
    NotAvailableError,
    SimulatedDataError,
)
from communitybridge.tools.distance import DistanceDataError, DistanceNotFoundError
from communitybridge.tools.logistics import LogisticsError
from communitybridge.tools.rule_access import RuleParameterError
from communitybridge.tools.service_query import ServiceQueryError
from communitybridge.trace.audit import (
    APPROVAL_TASK,
    BOOKING_TASK,
    BOOKING_TOOL,
    approval_id,
    latest_approval,
    latest_consent_approval,
    task_for,
)

# The tools' names in the permission map (`AGENT_TOOLS` in graph/build.py) and in the trace.
LOGISTICS_PLAN = "logistics_plan"
BOOKING = BOOKING_TOOL

GATE_THREE = "approve_booking"
_REASONS_SHOWN = 3


class LogisticsAgentError(Exception):
    """The graph asked this agent to do something it should never be asked to do (a bug)."""


class GateThreeAnswerError(LogisticsAgentError):
    """The answer sent back for Gate 3 was not one the gate understands."""


class ApprovalMissingError(LogisticsAgentError):
    """A booking was about to be made with no approval in the trace. It never is."""


class GateThreeAnswer(StrictModel):
    """What the interface sends back when the user answers Gate 3.

    `approve` names the option the user picked (a position in the list they were shown, from 0) and
    approves a simulated booking request for it. `decline` stops the run with no action taken.

    `contact` is the person's own details, given with an approval because that is the
    moment a booking needs them and one step before Gate 4 shows exactly what each action would
    share. It is checked by `ClientContact`, which refuses a record not marked synthetic (synthetic data only). The command line supplies a made-up contact when the run starts, so it is optional here.
    """

    decision: Literal["approve", "decline"]
    option: int | None = None
    contact: ClientContact | None = None

    @model_validator(mode="after")
    def _an_approval_names_its_option(self) -> Self:
        if self.decision == "approve" and (self.option is None or self.option < 0):
            raise ValueError("an approval needs the option chosen, counting from 0")
        if self.decision == "decline" and self.option is not None:
            raise ValueError("option is only for an approval")
        if self.decision == "decline" and self.contact is not None:
            raise ValueError("contact is only for an approval: nothing is done on a decline")
        return self


def parse_gate_three_answer(raw: object, option_count: int) -> GateThreeAnswer:
    """Check the answer from the interface. A bad one leaves the run waiting at the gate.

    The message names the fields at fault, never their values.
    """
    try:
        answer = GateThreeAnswer.model_validate(raw)
    except ValidationError as exc:
        raise GateThreeAnswerError(
            f"the answer to Gate 3 (approve booking) was not understood: {describe_problems(exc)}"
        ) from exc
    if answer.decision == "approve" and answer.option is not None and answer.option >= option_count:
        raise GateThreeAnswerError(
            "the answer to Gate 3 (approve booking) was not understood: "
            "option: must be one of the options offered"
        )
    return answer


def _event(task: str, source: str, result: str, status: Any) -> TraceEvent:
    return TraceEvent(
        task=task, agent="logistics", tool_or_source=source, result=result, status=status
    )


def _escalate(task: str, source: str, reason: str) -> dict[str, Any]:
    """End the run for staff. Two events: what failed, and the hand-over."""
    return {
        "status": "escalated",
        "trace": [
            _event(task, source, reason, "failed"),
            _event(
                "pass to staff",
                "escalation to staff",
                "the journey could not be planned or booked; a staff member can take the request",
                "escalated",
            ),
        ],
    }


# --- 1. planning ---------------------------------------------------------------------------------


# What the trace says about an attempt after the first plan. The first plan's text is
# unchanged.
_ATTEMPT_WORDS = {"next_window": "next appointment window: ", "telehealth": "telehealth: "}


def _plan_summary(plan: JourneyPlan, fallback: str = "initial") -> str:
    modes = plan.assessment.modes
    mode = modes.chosen_mode.replace("_", " ") if modes.chosen_mode else "depends on an answer"
    summary = _ATTEMPT_WORDS.get(fallback, "")
    summary += f"travel mode: {mode}; {len(plan.options)} option(s) fit the client's dates"
    if plan.rejected:
        summary += f"; {len(plan.rejected)} slot(s) did not fit"
    if plan.assessment.questions:
        summary += f"; {len(plan.assessment.questions)} question(s) for the client"
    if not plan.options:
        reasons = list(dict.fromkeys(r for w in plan.rejected for r in w.reasons))
        if reasons:
            summary += "; why: " + " ".join(reasons[:_REASONS_SHOWN])
    return summary


def make_plan_logistics(tools: ToolBox) -> NodeFn:
    """Build the plan_logistics node around the tool box for this agent."""

    @traced_node("logistics")
    def plan_logistics(state: GraphState) -> dict[str, Any]:
        goal = state.goal
        if goal is None:
            raise LogisticsAgentError("the state has no goal to plan a journey for")
        if not goal.confirmed_by_user:
            raise LogisticsAgentError("the goal has not been confirmed by the user (Gate 1)")
        if state.service is None or state.eligibility is None:
            raise LogisticsAgentError(
                "the state has no service and eligibility result to plan from"
            )
        if state.eligibility.outcome != "eligible":
            raise LogisticsAgentError("a journey is only planned for a client found eligible")
        task = "plan journey"
        # The attempt the Coordinator chose: the next window looks at the dates after the client's,
        # and only the telehealth attempt may ask for the specialist's telehealth slot.
        window = window_for(goal, state.fallback)
        search = goal.model_copy(
            update={"earliest_date": window.earliest_date, "latest_date": window.latest_date}
        )
        try:
            plan: JourneyPlan = tools.call(
                LOGISTICS_PLAN,
                search,
                state.facts,
                state.eligibility,
                state.service.service_id,
                state.fallback == "telehealth",
            )
        except LogisticsError:
            return _escalate(
                task, LOGISTICS_PLAN, "the request does not match the data the planner uses"
            )
        except DistanceNotFoundError:
            return _escalate(
                task, LOGISTICS_PLAN, "the dataset has no road distance for this journey"
            )
        except (DistanceDataError, RuleParameterError, SimulatedDataError):
            return _escalate(
                task, LOGISTICS_PLAN, "the rules, distance or simulated data could not be used"
            )
        except ServiceQueryError:
            return _escalate(task, LOGISTICS_PLAN, "the service database could not be read")
        status = "ok" if plan.options else "failed"
        return {
            "logistics": plan.assessment,
            "options": plan.options,
            "trace": [_event(task, LOGISTICS_PLAN, _plan_summary(plan, state.fallback), status)],
        }

    return plan_logistics


def route_after_planning(state: GraphState) -> Literal["verify", "replan", "finish"]:
    """Options go to the Verifier, then the user's approval of the plan, then Gate 3. None
    that fit go to re-planning. An escalated run finishes."""
    if state.status == "escalated":
        return "finish"
    if state.logistics is None:
        raise LogisticsAgentError("planning ended with no result and was not escalated")
    return "verify" if state.options else "replan"


# --- 2. Gate 3: the user approves a simulated booking request -------------------------------------


@traced_node("logistics")
def announce_approve_booking(state: GraphState) -> dict[str, Any]:
    """The node before the gate: records that the run is waiting for the user."""
    return {
        "status": "awaiting_approval",
        "trace": [
            _event(
                APPROVAL_TASK,
                "gate 3",
                "waiting for the user to choose an option and approve a simulated booking request",
                "awaiting_approval",
            )
        ],
    }


@traced_node("logistics")
def approve_booking(state: GraphState) -> dict[str, Any]:
    """Gate 3: show the options, wait, and act on the user's answer."""
    if not state.options:
        raise LogisticsAgentError("Gate 3 was reached with no options to approve")
    payload: dict[str, Any] = {
        "gate": GATE_THREE,
        "options": [option.model_dump(mode="json") for option in state.options],
    }
    while True:
        raw = interrupt(payload)
        try:
            answer = parse_gate_three_answer(raw, len(state.options))
            break
        except GateThreeAnswerError as exc:
            # As at Gate 1: LangGraph replays a stored answer, so raising here would trap the run.
            # Ask again, saying which field was wrong (names only, never the value).
            payload = {**payload, "error": str(exc)}

    if answer.decision == "approve":
        chosen = answer.option
        assert chosen is not None  # GateThreeAnswer guarantees an option for an approval
        # The details themselves go in the state and nowhere else: the trace says they were given
        # and never what they are (a name or a date of birth is about the person).
        given = {"contact": answer.contact} if answer.contact is not None else {}
        note = "; the user gave contact details" if answer.contact is not None else ""
        return {
            "chosen_option": chosen,
            **given,
            "status": "in_progress",
            "trace": [
                _event(
                    APPROVAL_TASK,
                    "user",
                    f"the user approved a simulated booking request for option {chosen + 1} of "
                    f"{len(state.options)}{note}",
                    "approved",
                )
            ],
        }
    return {
        "status": "declined",
        "trace": [
            _event(
                APPROVAL_TASK,
                "user",
                "the user declined; no booking request was made",
                "blocked",
            )
        ],
    }


def route_after_booking_gate(state: GraphState) -> Literal["consent", "finish"]:
    """Declined ends the run. An approval goes on to Gate 4, where the user decides, action by
    action, what data may be released. The booking step comes after that."""
    if state.status == "declined":
        return "finish"
    if state.chosen_option is None:
        raise LogisticsAgentError("Gate 3 ended with no choice and was not declined")
    return "consent"


# --- 3. the simulated booking request ------------------------------------------------------------


def make_request_booking(tools: ToolBox) -> NodeFn:
    """Build the request_booking node around the tool box for this agent."""

    @traced_node("logistics")
    def request_booking(state: GraphState) -> dict[str, Any]:
        if state.chosen_option is None or state.service is None:
            raise LogisticsAgentError("there is no chosen option to book")
        # The check that matters most in this module: nothing is requested unless the trace shows
        # the user approved it. It raises, so a missing approval is a bug that stops the run and
        # is never passed on as if it were a problem with the client's request.
        position = latest_approval(state.trace)
        if position is None:
            raise ApprovalMissingError(
                "no booking request is made without an approval in the trace"
            )
        approval = approval_id(position)
        option = state.options[state.chosen_option]
        # Only the kinds the user approved at Gate 4. A kind with no approval for its
        # action in the trace is a bug.
        asked = [
            KIND_FOR_ACTION[c.action]
            for c in state.consents
            if c.is_active and c.action in KIND_FOR_ACTION
        ]
        if not asked:
            raise LogisticsAgentError("no action was approved at Gate 4, so nothing is requested")
        for kind in asked:
            if latest_consent_approval(state.trace, ACTION_FOR_KIND[kind]) is None:
                raise ApprovalMissingError(
                    "no booking request is made for an action without its approval in the trace"
                )
        task = BOOKING_TASK
        try:
            made: list[BookingRequest] = tools.call(
                BOOKING, option, state.service.service_id, approval, asked
            )
        except (NotAvailableError, InvalidRequestError):
            return _escalate(
                task,
                BOOKING,
                "a part of the chosen option is no longer available; nothing was booked",
            )
        return {
            "bookings": made,
            "status": "planned",
            "trace": [
                _event(
                    task_for(BOOKING_TASK, ACTION_FOR_KIND[request.kind]),
                    BOOKING,
                    f"SIMULATED {request.kind} request {request.reference} recorded under "
                    f"approval {approval}",
                    "ok",
                )
                for request in made
            ],
        }

    return request_booking


def route_after_request(state: GraphState) -> Literal["release", "finish"]:
    """Requests that were made go on to the release of the data they needed. A request that
    could not be made ended the run for staff, so nothing is released for it."""
    if state.status == "escalated":
        return "finish"
    if not state.bookings:
        raise LogisticsAgentError("the booking step ended with no request and was not escalated")
    return "release"
