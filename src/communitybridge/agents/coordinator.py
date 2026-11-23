"""The Coordinator Agent: reads the user's goal and gets it confirmed at Gate 1.

The project design: the Coordinator "turns the goal into a structured request" and "the user
confirms the extracted details, then the plan". The project plan's first in-scope step is exactly
this: extract origin, specialty, dates and escort needs into a validated request that the user
confirms. This module is that first step. Re-planning (the Coordinator's other job) is handled further down.

Three design points worth being able to explain:

- **The model only fills in `ExtractedGoal`.** That is the six details without the
  `confirmed_by_user` flag, so a model can never confirm on the user's behalf. The model's reply
  must pass the schema (`llm/structured.generate`), and nothing it says can decide eligibility:
  its output has no such field.
- **Gate 1 is two nodes.** `interrupt()` pauses by raising, and the node that called it runs again
  from its first line on resume, its result saved only when it finishes. So one node
  records `awaiting_approval` first and a second asks the user and records the answer.
- **The model's identity is kept only if the person's own words state it**.
  `tools/identity.py` reads the goal text; a value it cannot find there becomes `not_stated`, with
  a trace note naming the field and not the value. The check only removes, so a statement it
  misses is put right by the user at Gate 1 or by the plan's identity question.
- **A model that fails sends the run to staff.** If the model is down, or keeps answering badly
  after the two allowed retries, the run ends as `escalated` with the reason in the trace. A guess
  is never passed on to the user as if it were what they said (the design diagram: escalate to staff).
- **The model chooses names from the project's lists, and a tool checks its choice.** A first live
  measurement found people say "heart specialist" where the rules engine needs "Cardiology". So the model is shown the specialty and town lists, and `tools/vocabulary.py` then
  verifies the pick: a match gets the listed spelling, and no match keeps the user's own words with
  a note. The lists are handed to this agent when the graph is built, as read-only reference data
  and not as a tool it calls, because the architecture table gives the Coordinator no tools
  (a small deviation from the original design).
- **Re-planning is bounded and follows fixed fallbacks**. When an attempt fails (planning
  finds no arrangement, the Verifier blocks a statement the plan rests on, or the user rejects the
  plan) the Coordinator re-plans along the set order and no other: the next appointment window, then
  telehealth, and after two failed re-plans it hands the run to staff. Telehealth is a last line of
  defence, so it is offered only when no in-person option is left. The node uses no tool
  and no model: it decides from the state, and the state itself refuses a fallback out of order.
- **Gate 2 is the user's approval of the plan**, the Coordinator's second approval point in
  the agent table. It shows the plan as the Verifier left it, blocked statements marked blocked, and
  asks again on a bad answer, like Gate 1. Approving does not book anything: that is Gate 3.
  If a language model wrote a wording of the decision, Gate 2 shows it too, apart from the plan and
  under a label that says what was and was not checked. It is never a plan item.

The user's own words are personal data. They are sent to the model as a user message and never
placed in the system message, a trace event or an error message.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Self

from langgraph.types import interrupt
from pydantic import ValidationError, model_validator

from communitybridge.graph.node_contract import NodeFn, describe_problems, traced_node
from communitybridge.graph.state import GraphState
from communitybridge.llm.base import LLMError, LLMOutputError, LLMProvider
from communitybridge.llm.structured import generate
from communitybridge.models import ExtractedGoal, GoalRequest, StrictModel, TraceEvent
from communitybridge.models.common import NonEmptyStr
from communitybridge.models.eligibility import MODEL_WORDING_LABEL
from communitybridge.models.facts import ConfirmedFacts
from communitybridge.models.llm import ChatMessage
from communitybridge.models.replan import (
    FALLBACK_ORDER,
    MAX_REPLANS,
    FailedAttempt,
    WhyFailed,
    window_for,
)
from communitybridge.tools.identity import check_identity
from communitybridge.tools.vocabulary import Vocabulary, resolve_goal

GATE_ONE = "confirm_details"

# The second version of the prompt. The first said "as they named it" for the town
# and specialty and had no lists; its live score was 79.4%. This one adds the project's own lists
# and clearer rules for dates, identity and escort, each answering a kind of error seen in that run.
SYSTEM_PROMPT = """\
You help a person in country South Australia who is planning to travel to see a medical \
specialist. Read their request and fill in the details they gave. Today's date is {today}.

Known specialties:
{specialties}

Known towns:
{towns}

Rules:
- Copy details only from what the person wrote. Never add, assume or guess a detail.
- origin_town: the town they are travelling from. If it is one of the known towns, write it \
exactly as listed. If not, write it in their own words, complete with any numbers or brackets.
- specialty: the kind of specialist or treatment they need. If it matches one of the known \
specialties, write that specialty exactly as listed. If none matches, write it in their own words.
- earliest_date and latest_date: the first and last day they could travel or attend, written as \
YYYY-MM-DD. If they give a particular date, a range of dates or a week, use exactly those dates \
(a week of a date is that date and the six days after it). Only if they name just a month, or say \
"next month", use the first and last day of that month; "next month" means the calendar month \
after today's. If they name only one day, use it for both dates.
- escort_requested: true only if they say someone will travel with them or that they need an \
escort; otherwise false. Staying with family or friends at the destination is not an escort.
- indigenous_status: aboriginal, torres_strait_islander, both or neither, only if the person says \
so in words themselves; otherwise not_stated. A name, a place, a job or a card (such as a Health \
Care Card) tells you nothing about it.
- Do not give advice and do not say whether they are eligible for, or qualify for, any payment or \
scheme. Someone else decides that.
- Reply with the JSON object only.\
"""


class CoordinatorError(Exception):
    """The Coordinator was asked to do something the graph should never ask of it."""


class GateAnswerError(CoordinatorError):
    """The answer sent back for Gate 1 was not one the gate understands."""


class GateOneAnswer(StrictModel):
    """What the interface sends back when the user answers Gate 1.

    `confirm` accepts the extracted details. `edit` gives new wording for the request, which is read
    again (the edge from confirm_details back to extract_goal in the state graph). `decline`
    stops the run with no action taken.

    `facts` is the whole set of details beyond the goal text as the person
    left them on the confirm page, with their corrections. It replaces the state's facts, and None
    leaves them as they were. It is a complete `ConfirmedFacts` and not a patch, so it is checked
    by the same model as the 30 scenarios' facts: an unknown or out-of-range value is refused, and
    there is no field in which an answer could carry a decision.
    """

    decision: Literal["confirm", "edit", "decline"]
    goal_text: NonEmptyStr | None = None
    facts: ConfirmedFacts | None = None

    @model_validator(mode="after")
    def _text_goes_with_an_edit_only(self) -> Self:
        if self.decision == "edit" and self.goal_text is None:
            raise ValueError("an edit needs the corrected goal_text")
        if self.decision != "edit" and self.goal_text is not None:
            raise ValueError("goal_text is only for an edit")
        if self.decision == "decline" and self.facts is not None:
            raise ValueError("facts are not for a decline: nothing is done")
        return self


def parse_gate_one_answer(raw: object) -> GateOneAnswer:
    """Check the answer from the interface. A bad one leaves the run waiting at the gate.

    The message names the fields at fault, never their values: an edit carries the user's words.
    """
    try:
        return GateOneAnswer.model_validate(raw)
    except ValidationError as exc:
        raise GateAnswerError(
            f"the answer to Gate 1 (confirm details) was not understood: {describe_problems(exc)}"
        ) from exc


def build_messages(goal_text: str, today: date, vocabulary: Vocabulary) -> list[ChatMessage]:
    """The request to the model: instructions as the system message, the user's words as data."""
    instructions = SYSTEM_PROMPT.format(
        today=today.isoformat(),
        specialties="\n".join(f"- {name}" for name in vocabulary.specialties),
        towns="\n".join(f"- {name}" for name in vocabulary.towns),
    )
    return [
        ChatMessage(role="system", content=instructions),
        ChatMessage(role="user", content=goal_text),
    ]


@dataclass(frozen=True)
class Extraction:
    """The checked result of one extraction: the details, the calls it took, and the notes."""

    goal: ExtractedGoal  # town and specialty already checked against the lists
    attempts: int  # LLM calls used, retries included
    notes: tuple[str, ...]  # fields kept as written because the lists have no match


def extract_goal_fields(
    provider: LLMProvider, goal_text: str, today: date, vocabulary: Vocabulary
) -> Extraction:
    """Ask the model for the six details, then check its town and specialty against the lists.

    The one place the prompt, the schema and the check are used, so the graph node and the accuracy
    measurement (scoring.py) test exactly the same thing.
    """
    generated = generate(provider, ExtractedGoal, build_messages(goal_text, today, vocabulary))
    resolution = resolve_goal(generated.value, vocabulary)
    goal, notes = resolution.goal, resolution.notes
    # Identity is the most sensitive field and the model has been seen to infer it from a place
    #. Keep its value only if the person's own words state it; the prompt is unchanged, so
    # the check is the only thing that differs from the earlier measurements.
    identity = check_identity(goal.indigenous_status, goal_text)
    if identity.removed:
        goal = goal.model_copy(update={"indigenous_status": identity.value})
        notes = (*notes, "indigenous_status set to not_stated (not found in the person's words)")
    return Extraction(goal=goal, attempts=generated.attempts, notes=notes)


def _event(task: str, source: str, result: str, status: Any) -> TraceEvent:
    return TraceEvent(
        task=task, agent="coordinator", tool_or_source=source, result=result, status=status
    )


def _escalate(source: str, reason: str) -> dict[str, Any]:
    """End the run for staff. Two events: what failed, and the hand-over."""
    return {
        "status": "escalated",
        "trace": [
            _event("extract goal", source, reason, "failed"),
            _event(
                "pass to staff",
                "escalation to staff",
                "no details were extracted; a staff member can take the request",
                "escalated",
            ),
        ],
    }


def make_extract_goal(
    provider: LLMProvider, vocabulary: Vocabulary, today: Callable[[], date] = date.today
) -> NodeFn:
    """Build the extract_goal node around a provider and the lists of known names. `today` is a
    function so a test or the evaluation can pin the date ("next month" depends on it)."""

    @traced_node("coordinator")
    def extract_goal(state: GraphState) -> dict[str, Any]:
        if state.goal_text is None:
            raise CoordinatorError("the state has no goal_text to extract details from")
        source = f"llm:{provider.name}"
        try:
            result = extract_goal_fields(provider, state.goal_text, today(), vocabulary)
        except LLMOutputError as exc:
            # Only the number of calls is recorded, never the reply: it may repeat the user's words.
            reason = f"the model's reply failed validation after {exc.attempts} calls"
            return _escalate(source, reason)
        except LLMError:
            reason = "the model could not be used (not running, or an unusable answer)"
            return _escalate(source, reason)
        calls = f"{result.attempts} LLM call" + ("" if result.attempts == 1 else "s")
        summary = f"details extracted in {calls}; not yet confirmed"
        if result.notes:
            # Field names only (see resolve_goal): the user's words never go into the trace.
            summary += "; " + "; ".join(result.notes)
        return {
            "goal": GoalRequest(**result.goal.model_dump()),  # confirmed_by_user stays False
            "trace": [_event("extract goal", source, summary, "ok")],
        }

    return extract_goal


@traced_node("coordinator")
def announce_confirm_details(state: GraphState) -> dict[str, Any]:
    """The node before the gate: records that the run is waiting for the user."""
    return {
        "status": "awaiting_approval",
        "trace": [
            _event(
                "confirm extracted details",
                "gate 1",
                "waiting for the user to confirm the extracted details",
                "awaiting_approval",
            )
        ],
    }


def _facts_changed(before: ConfirmedFacts, after: ConfirmedFacts) -> int:
    """How many of the facts differ, for the trace (a count, never the facts themselves)."""
    return sum(
        1 for name in ConfirmedFacts.model_fields if getattr(before, name) != getattr(after, name)
    )


@traced_node("coordinator")
def confirm_details(state: GraphState) -> dict[str, Any]:
    """Gate 1: show the user what was extracted, wait, and act on their answer."""
    if state.goal is None:
        raise CoordinatorError("Gate 1 was reached with no extracted goal")
    # The facts are shown too: the rules engine will use them, so an assumed value (the defaults
    # describe a typical adult client) is something the user can see and correct.
    payload: dict[str, Any] = {
        "gate": GATE_ONE,
        "goal": state.goal.model_dump(mode="json"),
        "facts": state.facts.model_dump(mode="json"),
    }
    while True:
        raw = interrupt(payload)
        try:
            answer = parse_gate_one_answer(raw)
            break
        except GateAnswerError as exc:
            # Raising here would trap the run: LangGraph stores an answer once it is given and
            # hands the same one back each time this node re-runs (found during testing). So ask
            # again instead. Each interrupt() call takes the next stored answer, and the "error"
            # field tells the interface what was wrong, in field names only.
            payload = {**payload, "error": str(exc)}

    # The corrected facts, if the person sent any. The trace says how many differ and never which
    # or what, because "has a concession card" or an age is about the person (like goal_text).
    corrections: dict[str, Any] = {}
    changed = 0
    if answer.facts is not None:
        corrections = {"facts": answer.facts}
        changed = _facts_changed(state.facts, answer.facts)
    changes_note = f"; the user changed {changed} of the assumed details" if changed else ""

    if answer.decision == "confirm":
        return {
            "goal": state.goal.model_copy(update={"confirmed_by_user": True}),
            **corrections,
            "status": "in_progress",
            "trace": [
                _event(
                    "confirm extracted details",
                    "user",
                    f"the user confirmed the extracted details{changes_note}",
                    "approved",
                )
            ],
        }
    if answer.decision == "edit":
        return {
            "goal": None,  # the old details are dropped; the new wording is read next
            "goal_text": answer.goal_text,
            **corrections,
            "status": "in_progress",
            "trace": [
                _event(
                    "edit details",
                    "user",
                    f"the user gave new wording; reading it again{changes_note}",
                    "ok",
                )
            ],
        }
    return {
        "status": "declined",
        "trace": [
            _event(
                "confirm extracted details",
                "user",
                "the user declined; nothing was done",
                "blocked",
            )
        ],
    }


def route_after_extract(state: GraphState) -> Literal["announce", "finish"]:
    """A failed extraction (escalated) ends the run; otherwise go to the gate."""
    return "finish" if state.status == "escalated" else "announce"


def route_after_confirm(state: GraphState) -> Literal["extract", "discover", "finish"]:
    """Declined ends the run; an edit (the goal was cleared) reads the new wording; confirmed goes
    on to Service Discovery."""
    if state.status == "declined":
        return "finish"
    if state.goal is None:
        return "extract"
    return "discover"


# --- Gate 2: the user approves the plan ---------------------------------------------------

GATE_TWO = "approve_plan"


class GateTwoAnswerError(CoordinatorError):
    """The answer sent back for Gate 2 was not one the gate understands."""


class GateTwoAnswer(StrictModel):
    """What the interface sends back when the user answers Gate 2.

    `approve` accepts the plan as shown (blocked statements stay shown as blocked). `reject` sends
    it back to be re-planned. A rejection counts as a failed attempt, so a
    user can never cause an endless loop.
    """

    decision: Literal["approve", "reject"]


def parse_gate_two_answer(raw: object) -> GateTwoAnswer:
    """Check the answer from the interface. A bad one leaves the run waiting at the gate."""
    try:
        return GateTwoAnswer.model_validate(raw)
    except ValidationError as exc:
        raise GateTwoAnswerError(
            f"the answer to Gate 2 (approve plan) was not understood: {describe_problems(exc)}"
        ) from exc


@traced_node("coordinator")
def announce_approve_plan(state: GraphState) -> dict[str, Any]:
    """The node before Gate 2: records that the run is waiting for the user."""
    return {
        "status": "awaiting_approval",
        "trace": [
            _event(
                "approve plan",
                "gate 2",
                "waiting for the user to approve the plan",
                "awaiting_approval",
            )
        ],
    }


def model_wording_for_gate(state: GraphState) -> dict[str, Any] | None:
    """The language model's wording of the decision, labelled, for the user to read beside the plan.

    None if the model wrote none (it was down, or failed its check). The label travels with the text
    so that no screen can show the wording without saying what it is.
    """
    if state.explanation is None:
        return None
    return {
        "label": MODEL_WORDING_LABEL,
        "text": state.explanation.text,
        "cited_sections": list(state.explanation.cited_sections),
    }


@traced_node("coordinator")
def approve_plan(state: GraphState) -> dict[str, Any]:
    """Gate 2: show the checked plan, wait, and act on the user's answer.

    The plan is shown exactly as the Verifier left it: every statement with its sources, and each
    blocked one marked blocked with the reason, plus the questions the user is asked to answer or
    confirm. Approving does not book anything: a booking needs Gate 3 as well.
    """
    if not state.plan:
        raise CoordinatorError("Gate 2 was reached with no plan to approve")
    payload: dict[str, Any] = {
        "gate": GATE_TWO,
        "plan": [item.model_dump(mode="json") for item in state.plan],
        "model_wording": model_wording_for_gate(state),
        "questions": list(state.questions),
        # True if approving leads on to choosing a booking (Gate 3); False for an information-only
        # plan, where approving ends the run.
        "booking_follows": bool(state.options),
        # Which attempt this plan is: the first plan, the next window, or telehealth.
        "fallback": state.fallback,
    }
    while True:
        raw = interrupt(payload)
        try:
            answer = parse_gate_two_answer(raw)
            break
        except GateTwoAnswerError as exc:
            # As at Gate 1: LangGraph replays a stored answer, so raising here would trap the run.
            payload = {**payload, "error": str(exc)}

    if answer.decision == "approve":
        return {
            "plan_approved": True,
            "status": "in_progress" if state.options else "planned",
            "trace": [
                _event(
                    "approve plan",
                    "user",
                    "the user approved the plan"
                    + ("" if state.options else " (information only: nothing to book)"),
                    "approved",
                )
            ],
        }
    return {
        "plan_approved": False,
        "status": "in_progress",
        "trace": [
            _event(
                "approve plan",
                "user",
                "the user rejected the plan; it goes back to be re-planned",
                "blocked",
            )
        ],
    }


def route_after_plan_gate(state: GraphState) -> Literal["book", "done", "replan"]:
    """A rejection goes back to be re-planned. An approval goes on to choose a booking
    (Gate 3) if there are options, and ends the run if the plan is information only."""
    if state.plan_approved is None:
        raise CoordinatorError("Gate 2 ended with no answer")
    if not state.plan_approved:
        return "replan"
    return "book" if state.options else "done"


# --- Re-planning: the next window, then telehealth, then staff ----------------------------

REPLAN_SOURCE = "predefined fallbacks"

_WHY_FAILED_WORDS: dict[str, str] = {
    "no_option": "planning found no arrangement that works",
    "blocked": "a statement the plan rests on was blocked",
    "rejected": "the plan was rejected",
}
_FALLBACK_WORDS: dict[str, str] = {
    "next_window": "the next appointment window",
    "telehealth": "telehealth, the specialist's own appointment",
}


def _why_failed(state: GraphState) -> WhyFailed:
    """Why the attempt under way has failed, read from the state. Reaching the re-plan step without
    a failure is a bug."""
    if state.plan_approved is False:
        return "rejected"
    if state.replan_needed:
        return "blocked"
    if (
        state.eligibility is not None
        and state.eligibility.outcome == "eligible"
        and not state.options
    ):
        return "no_option"
    raise CoordinatorError("the re-plan step was reached but nothing has failed")


def _in_person_left(state: GraphState, attempts: list[FailedAttempt]) -> str | None:
    """None if no in-person option is left, else the reason one is.

    Telehealth is a last line of defence. Every in-person window tried must have found nothing (a
    plan that was only rejected or blocked shows an in-person option exists), and the client must
    not be able to drive: a private vehicle (§7.2.1) is an in-person option no timetable can rule
    out.
    """
    if any(attempt.why != "no_option" for attempt in attempts):
        return "an in-person plan was made and then turned down or blocked"
    if state.facts.has_private_vehicle:
        return "the client has a private vehicle, which no timetable can rule out (7.2.1)"
    return None


def _give_up(attempts: list[FailedAttempt], reason: str) -> dict[str, Any]:
    """End the run for staff. Two events: why no further fallback is tried, and the hand-over."""
    return {
        "status": "escalated",
        "attempts": attempts,
        "trace": [
            _event("re-plan", REPLAN_SOURCE, reason, "failed"),
            _event(
                "pass to staff",
                "escalation to staff",
                "no further fallback is tried; a staff member can take the request",
                "escalated",
            ),
        ],
    }


@traced_node("coordinator")
def replan(state: GraphState) -> dict[str, Any]:
    """An attempt has failed: re-plan along the next fallback, or give the run to staff.

    Design: the fallbacks are fixed and in one order, "the next appointment window, then telehealth",
    and after two failed re-plans the run goes to staff. Each failure counts as one
    attempt whatever caused it, so a user who keeps rejecting plans cannot cause an endless loop.
    Telehealth is offered only if no in-person option is left, and the check is recorded in the
    trace first. A client who is not eligible has a plan that is only information, and
    re-planning never changes the PATS decision, so there is nothing to try and a person takes over.

    A re-plan clears the old plan and everything the user answered about it, so the new plan is
    judged and approved afresh. Nothing downstream of the plan exists yet, so nothing else needs
    undoing.
    """
    goal = state.goal
    if goal is None or not goal.confirmed_by_user:
        raise CoordinatorError("there is no confirmed goal to re-plan for")
    if state.service is None or state.eligibility is None:
        raise CoordinatorError("the state has no service and eligibility result to re-plan from")
    why = _why_failed(state)
    count = state.replan_count
    attempts = [
        *state.attempts,
        FailedAttempt(fallback=state.fallback, window=window_for(goal, state.fallback), why=why),
    ]
    if state.eligibility.outcome != "eligible":
        return _give_up(
            attempts, "the plan only gives information, so there is no other arrangement to try"
        )
    if count >= MAX_REPLANS:
        return _give_up(
            attempts, "both re-plans (the next appointment window, then telehealth) have been used"
        )

    next_fallback = FALLBACK_ORDER[count + 1]
    events = []
    if next_fallback == "telehealth":
        left = _in_person_left(state, attempts)
        if left is not None:
            return _give_up(
                attempts, f"telehealth is not offered while an in-person option is left: {left}"
            )
        events.append(
            _event(
                "check no in-person option is left",
                REPLAN_SOURCE,
                f"every in-person window tried ({len(attempts)} windows) found no arrangement "
                "and the client has no private vehicle (7.2.1), so telehealth may be offered",
                "ok",
            )
        )
    events.append(
        _event(
            "re-plan",
            REPLAN_SOURCE,
            f"re-plan {count + 1} of {MAX_REPLANS}: {_FALLBACK_WORDS[next_fallback]}, because "
            f"{_WHY_FAILED_WORDS[why]}",
            "ok",
        )
    )
    return {
        "replan_count": count + 1,
        "fallback": next_fallback,
        "attempts": attempts,
        "status": "in_progress",
        # The old plan, and the answers to it, are cleared before the new attempt.
        "logistics": None,
        "options": [],
        "plan": [],
        "questions": [],
        "replan_needed": False,
        "plan_approved": None,
        "trace": events,
    }


def route_after_replan(state: GraphState) -> Literal["plan", "finish"]:
    """A re-plan goes back to planning the journey. An escalated run finishes."""
    return "finish" if state.status == "escalated" else "plan"
