"""All the web pages and the forms that answer the four gates.

The goal page and Gate 1 (confirm the details), the plan and Gate 2, the booking choice and Gate 3,
the consent screen and Gate 4 with withdrawal, the agent trace, and the escalation message. They
follow the design diagram: the person answers each gate in turn and nothing happens without an answer.

Routes are plain `def`, not `async def`, so FastAPI runs them in a worker thread. Running the graph
can take seconds (a model call), and an `async def` route would hold up every other request.

Every answer is a POSTed form. Opening a page, however often, never answers a gate; the server
records an answer only when a valid form arrives for the gate that is waiting.
"""

from collections.abc import Mapping
from functools import cache
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from communitybridge.agents.consent import GATE_FOUR
from communitybridge.agents.coordinator import (
    GATE_ONE,
    GATE_TWO,
    GateAnswerError,
    model_wording_for_gate,
    parse_gate_one_answer,
)
from communitybridge.agents.logistics import GATE_THREE
from communitybridge.tools.policy_loader import load_consent_policy
from communitybridge.web.forms import (
    DECISIONS,
    GOAL_MAX_CHARS,
    check_goal_text,
    parse_booking_answer,
    parse_consent_answer,
    parse_fact_answers,
    parse_plan_decision,
)
from communitybridge.web.journeys import JourneyService, JourneyView
from communitybridge.web.present import (
    EDITABLE_FACTS,
    assumed_rows,
    consent_cards,
    controls,
    escalation_context,
    format_date,
    goal_rows,
    option_cards,
    outcome_context,
    plan_page_context,
    trace_rows,
)

# The page for each question a journey can be waiting on, so one place decides where a person goes.
GATE_PAGES = {
    GATE_ONE: "confirm",
    GATE_TWO: "plan",
    GATE_THREE: "booking",
    GATE_FOUR: "consent",
}


class GateMovedOnError(Exception):
    """The person answered a question the journey is no longer waiting on."""


class PlanNotReadyError(Exception):
    """The journey has not got as far as a plan yet."""


class BadFormError(Exception):
    """A form was posted without something every form of its kind carries (a script, not a page)."""


@cache
def withdrawal_text() -> dict[str, str]:
    """What withdrawing each action undoes, in the consent policy's own words (read once)."""
    return {name: action.on_withdrawal for name, action in load_consent_policy().actions.items()}


async def posted_fields(request: Request) -> dict[str, str]:
    """The text fields of a POSTed form, with "" kept as "" and a missing field left out.

    FastAPI's own `Form(None)` parameters turn an empty box into "not sent", so a person who clears
    the age box would look the same as one who never saw it, and the fact would silently stay at
    its assumed value. Reading the form here keeps the difference, so an empty box can be refused.
    (An `async` dependency, so the route itself can stay a plain `def` run in a worker thread.)
    """
    form = await request.form()
    return {name: value for name, value in form.items() if isinstance(value, str)}


Fields = Depends(posted_fields)


def _see_other(url: str) -> RedirectResponse:
    # 303 See Other: the browser follows with a GET, so refreshing never posts the form again.
    return RedirectResponse(url, status_code=303)


def build_router(service: JourneyService, templates: Jinja2Templates) -> APIRouter:
    router = APIRouter()

    def goal_page(
        request: Request, text: str = "", problems: list[str] | None = None
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "goal.html",
            {
                "nav_current": "home",
                "goal_text": text,
                "problems": problems or [],
                "reference_date": format_date(service.reference_today()),
                "max_chars": GOAL_MAX_CHARS,
            },
            status_code=422 if problems else 200,
        )

    def confirm_page(
        request: Request,
        journey_id: str,
        view: JourneyView,
        submitted: dict[str, str | None] | None = None,
        problems: list[str] | None = None,
        wording: str | None = None,
    ) -> HTMLResponse:
        if view.gate is None or view.gate.get("gate") != GATE_ONE:
            raise GateMovedOnError(journey_id)
        submitted = submitted or {}
        return templates.TemplateResponse(
            request,
            "confirm.html",
            {
                "nav_current": "",
                "journey_id": journey_id,
                "goal_rows": goal_rows(view.gate["goal"]),
                "assumed_rows": assumed_rows(view.gate["facts"]),
                "controls": controls(view.state.facts.model_dump(), submitted),
                "wording": view.state.goal_text if wording is None else wording,
                "own_words": view.state.goal_text,
                "problems": problems or [],
                "max_chars": GOAL_MAX_CHARS,
            },
            status_code=422 if problems else 200,
        )

    @router.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return goal_page(request)

    @router.post("/goal", response_class=HTMLResponse)
    def start_journey(request: Request, fields: dict[str, str] = Fields) -> Response:
        typed = fields.get("goal_text", "")
        text, problem = check_goal_text(typed)
        if problem:
            return goal_page(request, typed, [problem])
        journey_id, view = service.start(text)
        at_gate_one = view.gate is not None and view.gate.get("gate") == GATE_ONE
        where = f"/journeys/{journey_id}" + ("/confirm" if at_gate_one else "")
        return _see_other(where)

    @router.get("/journeys/{journey_id}/confirm", response_class=HTMLResponse)
    def confirm_form(request: Request, journey_id: str) -> HTMLResponse:
        return confirm_page(request, journey_id, service.view(journey_id))

    @router.post("/journeys/{journey_id}/confirm", response_class=HTMLResponse)
    def confirm_answer(
        request: Request, journey_id: str, fields: dict[str, str] = Fields
    ) -> Response:
        view = service.view(journey_id)
        if view.gate is None or view.gate.get("gate") != GATE_ONE:
            raise GateMovedOnError(journey_id)

        decision = fields.get("decision", "")
        goal_text = fields.get("goal_text", "")
        # Only the facts the page offers are read: the form can set nothing else.
        submitted: dict[str, str | None] = {name: fields.get(name) for name in EDITABLE_FACTS}

        problems: list[str] = []
        if decision not in DECISIONS:
            problems.append("Please choose one of the three buttons: confirm, change, or stop.")
        wording = ""
        if decision == "edit":
            wording, wording_problem = check_goal_text(goal_text)
            if wording_problem:
                problems.append(wording_problem)
        facts, fact_problems = (
            (None, []) if decision == "decline" else parse_fact_answers(view.state.facts, submitted)
        )
        problems.extend(fact_problems)
        if problems:
            return confirm_page(request, journey_id, view, submitted, problems, goal_text)

        raw: dict[str, Any] = {"decision": decision}
        if decision == "edit":
            raw["goal_text"] = wording
        if facts is not None:
            raw["facts"] = facts.model_dump(mode="json")
        try:
            parse_gate_one_answer(raw)  # the same check the graph makes; refuse here, not there
        except GateAnswerError:
            return confirm_page(
                request, journey_id, view, submitted, ["Your answer could not be used."], goal_text
            )
        service.answer(journey_id, raw)
        return _see_other(f"/journeys/{journey_id}")

    def pending(view: JourneyView, gate: str, journey_id: str) -> Mapping[str, Any]:
        """The payload of the question the journey is waiting on, if it is `gate`; else a 409.

        Every page that asks a question and every POST that answers one goes through this, so an
        answer can only ever reach the gate that is actually waiting: a Gate 4 form posted at Gate 2
        is refused and changes nothing.
        """
        if view.gate is None or view.gate.get("gate") != gate:
            raise GateMovedOnError(journey_id)
        return view.gate

    # --- Gate 2: approve the plan ----------------------------------------------------------

    def plan_page(
        request: Request,
        journey_id: str,
        view: JourneyView,
        problems: list[str] | None = None,
    ) -> HTMLResponse:
        state = view.state
        if not state.plan:
            raise PlanNotReadyError(journey_id)
        waiting = view.gate is not None and view.gate.get("gate") == GATE_TWO
        return templates.TemplateResponse(
            request,
            "plan.html",
            {
                "nav_current": "",
                "journey_id": journey_id,
                **plan_page_context(state),
                # The language model's wording, labelled and apart from the plan. The
                # same function the Gate 2 payload uses, so the web cannot word the label itself.
                "wording": model_wording_for_gate(state),
                "awaiting_decision": waiting,
                "booking_follows": bool(view.gate and view.gate.get("booking_follows")),
                "problems": problems or [],
            },
            status_code=422 if problems else 200,
        )

    @router.get("/journeys/{journey_id}/plan", response_class=HTMLResponse)
    def plan(request: Request, journey_id: str) -> HTMLResponse:
        return plan_page(request, journey_id, service.view(journey_id))

    @router.post("/journeys/{journey_id}/plan", response_class=HTMLResponse)
    def plan_answer(request: Request, journey_id: str, fields: dict[str, str] = Fields) -> Response:
        view = service.view(journey_id)
        pending(view, GATE_TWO, journey_id)
        raw, problems = parse_plan_decision(fields)
        if raw is None:
            return plan_page(request, journey_id, view, problems)
        service.answer(journey_id, raw)
        return _see_other(f"/journeys/{journey_id}")

    # --- Gate 3: choose a booking to ask for, and give contact details ------------------------

    def booking_page(
        request: Request,
        journey_id: str,
        view: JourneyView,
        values: Mapping[str, str] | None = None,
        problems: list[str] | None = None,
    ) -> HTMLResponse:
        gate = pending(view, GATE_THREE, journey_id)
        return templates.TemplateResponse(
            request,
            "booking.html",
            {
                "nav_current": "",
                "journey_id": journey_id,
                "cards": option_cards(gate["options"]),
                "values": dict(values or {}),
                "problems": problems or [],
            },
            status_code=422 if problems else 200,
        )

    @router.get("/journeys/{journey_id}/booking", response_class=HTMLResponse)
    def booking(request: Request, journey_id: str) -> HTMLResponse:
        return booking_page(request, journey_id, service.view(journey_id))

    @router.post("/journeys/{journey_id}/booking", response_class=HTMLResponse)
    def booking_answer(
        request: Request, journey_id: str, fields: dict[str, str] = Fields
    ) -> Response:
        view = service.view(journey_id)
        gate = pending(view, GATE_THREE, journey_id)
        raw, problems = parse_booking_answer(
            fields, len(gate["options"]), service.reference_today()
        )
        if raw is None:
            return booking_page(request, journey_id, view, fields, problems)
        service.answer(journey_id, raw)
        return _see_other(f"/journeys/{journey_id}")

    # --- Gate 4: what may be shared, action by action -----------------------------------------

    def consent_page(
        request: Request,
        journey_id: str,
        view: JourneyView,
        chosen: Mapping[str, str] | None = None,
        problems: list[str] | None = None,
    ) -> HTMLResponse:
        gate = pending(view, GATE_FOUR, journey_id)
        return templates.TemplateResponse(
            request,
            "consent.html",
            {
                "nav_current": "",
                "journey_id": journey_id,
                "cards": consent_cards(gate["actions"], chosen or {}),
                "problems": problems or [],
            },
            status_code=422 if problems else 200,
        )

    @router.get("/journeys/{journey_id}/consent", response_class=HTMLResponse)
    def consent(request: Request, journey_id: str) -> HTMLResponse:
        return consent_page(request, journey_id, service.view(journey_id))

    @router.post("/journeys/{journey_id}/consent", response_class=HTMLResponse)
    def consent_answer(
        request: Request, journey_id: str, fields: dict[str, str] = Fields
    ) -> Response:
        view = service.view(journey_id)
        gate = pending(view, GATE_FOUR, journey_id)
        # Only the actions the page offered are read; any other field in the POST is ignored.
        offered = {prompt["action"]: prompt["description"] for prompt in gate["actions"]}
        raw, problems = parse_consent_answer(fields, offered)
        if raw is None:
            return consent_page(request, journey_id, view, fields, problems)
        service.answer(journey_id, raw)
        return _see_other(f"/journeys/{journey_id}")

    # --- withdrawing consent after the run -------------------------------------------

    @router.post("/journeys/{journey_id}/withdraw", response_class=HTMLResponse)
    def withdraw(request: Request, journey_id: str, fields: dict[str, str] = Fields) -> Response:
        action = fields.get("action", "").strip()
        if not action:
            raise BadFormError("Please choose which consent to withdraw.")
        service.withdraw(journey_id, action)
        return _see_other(f"/journeys/{journey_id}")

    # --- the agent trace ---------

    @router.get("/journeys/{journey_id}/trace", response_class=HTMLResponse)
    def trace(request: Request, journey_id: str) -> HTMLResponse:
        # Read-only at every stage of a journey: it shows what has been recorded so far and can
        # answer nothing, so it needs no gate check.
        state = service.view(journey_id).state
        return templates.TemplateResponse(
            request,
            "trace.html",
            {"nav_current": "trace", "journey_id": journey_id, "rows": trace_rows(state.trace)},
        )

    # --- where a journey is up to ----------------------------------------------------------

    @router.get("/journeys/{journey_id}", response_class=HTMLResponse)
    def journey(request: Request, journey_id: str) -> Response:
        view = service.view(journey_id)
        gate = None if view.gate is None else view.gate.get("gate")
        if gate in GATE_PAGES:
            return _see_other(f"/journeys/{journey_id}/{GATE_PAGES[gate]}")
        state = view.state
        if state.status == "escalated":
            return templates.TemplateResponse(
                request,
                "escalated.html",
                {"nav_current": "", "journey_id": journey_id, **escalation_context(state)},
            )
        return templates.TemplateResponse(
            request,
            "journey.html",
            {
                "nav_current": "",
                "journey_id": journey_id,
                "status": state.status,
                "confirmed": state.goal is not None and state.goal.confirmed_by_user,
                "has_plan": bool(state.plan),
                **outcome_context(state, withdrawal_text()),
            },
        )

    return router
