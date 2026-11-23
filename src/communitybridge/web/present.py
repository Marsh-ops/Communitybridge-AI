"""Plain-language labels and formatting for the confirm page and the plan view.

Serves Gate 1 (the design diagram: "the user confirms the extracted details"). The person confirming is
a community member, not a developer, so every detail is shown under a plain label and never under
its code name (`has_private_vehicle`). a test checks that every goal field
and every fact in the models has a label, so adding a field later fails a test until it is worded.

Which facts a person can correct on the page (`EDITABLE_FACTS`) is a design decision:
- Five are plain questions a client can answer about themselves: permanent residence in South
  Australia, age, concession card, staying with family or friends, and having a car.
- The rest are shown but not editable here. Some carry a defined meaning in the PATS Guidelines
  (severe financial hardship, a specialist's authorisation, the reason for travelling past a
  nearer service, the distance of an alternative route). Asking about them in different words would
  put words in the Guidelines' mouth, so they stay as the scenarios or a staff member
  set them until the PDF's wording is used for each question.

The rules engine still decides everything: these are inputs it is given, never decisions.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal

from communitybridge.graph.state import GraphState
from communitybridge.models.common import Citation
from communitybridge.models.plan import PlanItem
from communitybridge.models.trace import TraceEvent

GOAL_LABELS = {
    "origin_town": "Travelling from",
    "specialty": "Type of specialist",
    "earliest_date": "Earliest date you can go",
    "latest_date": "Latest date you can go",
    "escort_requested": "Someone travelling with you",
    "indigenous_status": "Aboriginal or Torres Strait Islander (only if you said so)",
}

# Every fact in `ConfirmedFacts`, in the order the model lists them.
FACT_LABELS = {
    "indigenous_status": "Identity used by the rules",
    "permanent_sa_resident": "Lives permanently in South Australia",
    "residency_situation": "Residency situation",
    "away_from_residence": "Travelling from somewhere other than home",
    "client_age": "Age",
    "concession_card": "Concession card",
    "severe_financial_hardship": "Severe financial hardship",
    "notice_business_days": "Business days of notice before travel",
    "appointment_type": "Type of appointment",
    "treatment": "Treatment",
    "subspecialty": "Subspecialty",
    "alternate_route_km": "Length of the alternative route (km)",
    "bypass_reason": "Reason for travelling past a nearer service",
    "gp_reason_documented": "Referring practitioner has written down the reason",
    "local_timeframe_clinically_acceptable": "Waiting for a local appointment is clinically fine",
    "visiting_primary_rooms_service_id": "Visiting specialist's main rooms service",
    "referred_from_tertiary_site": "Referred from a SA Health tertiary site",
    "escort_criterion": "Reason an escort is needed",
    "escort_specialist_approved": "Specialist has approved an escort",
    "air_medical_criterion": "Medical reason for air travel",
    "air_specialist_authorised": "Specialist has authorised air travel",
    "requested_modes": "Ways of travelling asked for",
    "has_private_vehicle": "Has a car to use",
    "staying_with_family_or_friends": "Staying with family or friends",
    "specialist_authorised_nights": "Nights authorised by the specialist",
}

# Words for the values people read, where the stored value is a code.
_IDENTITY_WORDS = {
    "aboriginal": "Aboriginal",
    "torres_strait_islander": "Torres Strait Islander",
    "both": "Aboriginal and Torres Strait Islander",
    "neither": "Neither Aboriginal nor Torres Strait Islander",
    "not_stated": "Not stated",
}


@dataclass(frozen=True)
class EditableFact:
    """A fact the person can correct on the confirm page, and the question that asks for it."""

    question: str
    kind: Literal["yes_no", "whole_number"]


EDITABLE_FACTS = {
    "permanent_sa_resident": EditableFact("Do you live permanently in South Australia?", "yes_no"),
    "client_age": EditableFact("How old are you?", "whole_number"),
    "concession_card": EditableFact("Do you hold a concession card?", "yes_no"),
    "staying_with_family_or_friends": EditableFact(
        "Will you stay with family or friends instead of paid accommodation?", "yes_no"
    ),
    "has_private_vehicle": EditableFact("Do you have a car you could use for the trip?", "yes_no"),
}

AGE_RANGE = (0, 120)


def format_date(value: date) -> str:
    """A date as people say it: 1 November 2026. (`%-d` is not available on Windows.)"""
    return f"{value.day} {value:%B %Y}"


def format_value(value: object) -> str:
    """A stored value in words a person reads."""
    if value is None:
        return "Not stated"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "None"
    text = str(value)
    if text in _IDENTITY_WORDS:
        return _IDENTITY_WORDS[text]
    if "_" in text:
        text = text.replace("_", " ")
        return text[:1].upper() + text[1:]
    return text


@dataclass(frozen=True)
class Row:
    """One line of a definition list: a label, the words, and an ISO date for `<time>` if any."""

    label: str
    text: str
    iso: str | None = None


def goal_rows(goal: Mapping[str, Any]) -> list[Row]:
    """The six extracted details, in the order the page shows them."""
    rows = []
    for name, label in GOAL_LABELS.items():
        value = goal[name]
        if name.endswith("_date"):
            parsed = date.fromisoformat(str(value))
            rows.append(Row(label, format_date(parsed), parsed.isoformat()))
        else:
            rows.append(Row(label, format_value(value)))
    return rows


def assumed_rows(facts: Mapping[str, Any]) -> list[Row]:
    """The facts a person cannot change on this page, in words."""
    rows = []
    for name, label in FACT_LABELS.items():
        if name in EDITABLE_FACTS:
            continue
        if name == "indigenous_status" and facts[name] is None:
            # None means "use what the request says" (models/facts.py), not "unknown".
            rows.append(Row(label, "Taken from your request"))
        else:
            rows.append(Row(label, format_value(facts[name])))
    return rows


@dataclass(frozen=True)
class Control:
    """One editable fact as the template draws it, with the answer to preselect."""

    name: str
    question: str
    kind: Literal["yes_no", "whole_number"]
    value: str  # "yes" / "no" for a yes_no question, the number as text for a whole number


def controls(facts: Mapping[str, Any], submitted: Mapping[str, str | None]) -> list[Control]:
    """The editable questions. What the person typed (after a refusal) wins over the saved fact."""
    result = []
    for name, spec in EDITABLE_FACTS.items():
        typed = submitted.get(name)
        if typed is not None:
            value = typed
        elif spec.kind == "yes_no":
            value = "yes" if facts[name] else "no"
        else:
            value = str(facts[name])
        result.append(Control(name, spec.question, spec.kind, value))
    return result


# --- the plan view --------------------------------------------------------------------


@dataclass(frozen=True)
class SourceGroup:
    """One source with the date it was last checked and every section or record cited from it."""

    source: str
    verified_text: str
    verified_iso: str
    sections: list[str]


@dataclass(frozen=True)
class PlanRow:
    """One plan statement as the page draws it."""

    number: int
    statement: str
    verified: bool
    simulated: bool
    sources: list[SourceGroup]
    blocked_reason: str | None


def is_simulated(item: PlanItem) -> bool:
    """Does this statement rest on a made-up timetable, fare sheet or calendar?

    The badge is the core rule ("the UI shows a visible SIMULATED badge"). Two signs are used because
    the plan builder gives two: an option cites the simulated service's own source ("SIMULATED
    transport timetable ..."), and a statement worked out from the fare sheet says "(SIMULATED
    fares)" in its own words while citing the PATS sections it applies. Either sign means badge, so
    a statement can never carry the words without the badge. A statement shown without a badge is
    one the plan builder gave neither sign.
    """
    if "simulated" in item.statement.lower():
        return True
    return any("simulated" in citation.source.lower() for citation in item.citations)


def group_sources(citations: Sequence[Citation]) -> list[SourceGroup]:
    """One entry per source and verification date, listing each section once (as the CLI does).

    A statement can rest on a dozen PATS sections; a line each would bury it. The source name, the
    date it was last verified and every section are all still shown, so a person can see the
    Verifier's freshness check, not only be told it was done.
    """
    grouped: dict[tuple[str, date], list[str]] = {}
    for citation in citations:
        sections = grouped.setdefault((citation.source, citation.last_verified), [])
        if citation.section_or_record not in sections:
            sections.append(citation.section_or_record)
    return [
        SourceGroup(source, format_date(verified), verified.isoformat(), sections)
        for (source, verified), sections in grouped.items()
    ]


def plan_rows(plan: Sequence[PlanItem]) -> list[PlanRow]:
    """The plan statements in order, numbered from 1, as the template draws them."""
    return [
        PlanRow(
            number=number,
            statement=item.statement,
            verified=item.status == "verified",
            simulated=is_simulated(item),
            sources=group_sources(item.citations),
            blocked_reason=item.blocked_reason,
        )
        for number, item in enumerate(plan, start=1)
    ]


_FALLBACK_NOTES = {
    "next_window": (
        "This is a new plan. None of the dates you asked for worked, so it uses the next "
        "appointment window."
    ),
    "telehealth": (
        "This is a new plan. No in-person arrangement worked, so it is a telehealth appointment."
    ),
}


def fallback_note(fallback: str) -> str | None:
    """What a re-planned plan says about itself; the first plan says nothing."""
    return _FALLBACK_NOTES.get(fallback)


def plan_page_context(state: GraphState) -> dict[str, Any]:
    """What the plan page shows, worked out from the graph state alone.

    Kept apart from the route so each part can be tested without running a journey: the real graph
    never leaves a blocked statement, a question or a re-plan at Gate 2 for the Ceduna scenario,
    but the page must still show them correctly when another scenario does.
    """
    rows = plan_rows(state.plan)
    verified = sum(1 for row in rows if row.verified)
    return {
        "rows": rows,
        "verified": verified,
        "blocked": len(rows) - verified,
        "questions": list(state.questions),
        "fallback_note": fallback_note(state.fallback),
    }


# --- Gates 3 and 4, and the result ------------------------------------------------------


def format_datetime(iso: str) -> str:
    """An ISO date and time as a person says it: Tuesday 10 November 2026, 9:30 am."""
    value = datetime.fromisoformat(iso)
    hour = value.hour % 12 or 12
    am_pm = "am" if value.hour < 12 else "pm"
    return f"{value:%A} {value.day} {value:%B %Y}, {hour}:{value:%M} {am_pm}"


def format_money(amount: float | None) -> str:
    return "not worked out" if amount is None else f"${amount:,.2f}"


@dataclass(frozen=True)
class OptionCard:
    """One booking option for Gate 3, as the page draws it. Every option is SIMULATED."""

    value: str  # the position the graph counts from 0, which is what the form sends back
    number: int  # counted from 1, which is what the person reads
    rows: list[Row]
    notes: list[str]
    sources: list[SourceGroup]


def option_cards(options: Sequence[Mapping[str, Any]]) -> list[OptionCard]:
    """The options Gate 3 offers, in the order the graph holds them."""
    cards = []
    for index, option in enumerate(options):
        mode = format_value(option["travel_mode"])
        rows = [
            Row("Travel", f"{mode[:1].upper() + mode[1:]}, {option['travellers']} traveller(s)")
        ]
        if option.get("outbound_date") and option.get("return_date"):
            go = date.fromisoformat(option["outbound_date"])
            back = date.fromisoformat(option["return_date"])
            rows.append(Row("Go", format_date(go), go.isoformat()))
            rows.append(Row("Return", format_date(back), back.isoformat()))
        nights = option.get("nights", 0)
        if nights and option.get("accommodation_name"):
            rows.append(Row("Stay", f"{nights} night(s) at {option['accommodation_name']}"))
        else:
            rows.append(Row("Stay", "No overnight stay"))
        rows.append(Row("Appointment", format_datetime(option["appointment_start"])))
        rows.append(
            Row(
                "PATS would pay up to",
                f"travel {format_money(option.get('travel_subsidy_aud'))}, "
                f"accommodation {format_money(option.get('accommodation_subsidy_aud'))}",
            )
        )
        cards.append(
            OptionCard(
                value=str(index),
                number=index + 1,
                rows=rows,
                notes=list(option.get("notes", [])),
                sources=group_sources([Citation.model_validate(c) for c in option["citations"]]),
            )
        )
    return cards


def consent_cards(
    prompts: Sequence[Mapping[str, Any]], chosen: Mapping[str, str]
) -> list[dict[str, Any]]:
    """The Gate 4 prompts as cards: what each action shares, with whom, and the answer so far.

    `chosen` holds answers already given (after a refused form), so nothing is retyped. No answer
    is ever pre-selected otherwise: a person has to say approve or decline for each action.
    """
    cards = []
    for prompt in prompts:
        action = prompt["action"]
        cards.append(
            {
                "action": action,
                "description": prompt["description"],
                "recipient": prompt["recipient"],
                "purpose": prompt["purpose"],
                "on_withdrawal": prompt["on_withdrawal"],
                "missing": list(prompt.get("missing", [])),
                "fields": [
                    {"name": f["name"], "description": f["description"], "value": f.get("value")}
                    for f in prompt["fields"]
                ],
                "field_name": f"decision-{action}",
                "chosen": chosen.get(f"decision-{action}", ""),
            }
        )
    return cards


_KIND_WORDS = {"appointment": "Appointment", "transport": "Transport", "accommodation": "Stay"}


def outcome_context(state: GraphState, withdrawal_text: Mapping[str, str]) -> dict[str, Any]:
    """What the result page shows after the last gate: the SIMULATED requests, what each recipient
    was given (field names only, never the values), what was withdrawn, and what can still be.

    `withdrawal_text` maps an action to the policy's words about what withdrawing undoes.
    """
    withdrawn = {notice.action for notice in state.withdrawals}
    return {
        "bookings": [
            {
                "kind": _KIND_WORDS.get(b.kind, b.kind),
                "reference": b.reference,
                "status": b.status,
                "description": b.description,
            }
            for b in state.bookings
        ],
        "shared": [
            {
                "recipient": r.release.recipient,
                "fields": [name.replace("_", " ") for name in r.release.data],
            }
            for r in state.releases
        ],
        "withdrawals": [
            {"recipient": n.recipient, "message": n.message} for n in state.withdrawals
        ],
        "can_withdraw": [
            {
                "action": r.release.action,
                "recipient": r.release.recipient,
                "text": withdrawal_text.get(r.release.action, ""),
            }
            for r in state.releases
            if r.release.action not in withdrawn
        ],
    }


# --- the agent trace ----------------------------------------------------------------------

# Plain words for the fixed lists in `models/trace.py`. A test checks every agent and every status
# has an entry, so adding one to the model fails until it is worded.
AGENT_WORDS = {
    "coordinator": "Coordinator",
    "service_discovery": "Service discovery",
    "eligibility": "Eligibility",
    "logistics": "Logistics",
    "verifier": "Verifier",
    "consent": "Consent",
}
STATUS_WORDS = {
    "ok": "Done",
    "approved": "Approved by you",
    "blocked": "Blocked",
    "failed": "Failed",
    "awaiting_approval": "Waiting for you",
    "withdrawn": "Withdrawn",
    "escalated": "Passed to staff",
}
# Which badge style a status gets. The word is always shown as well, so colour is never the only
# sign (WCAG 1.4.1).
_STATUS_STYLE = {
    "ok": "badge-verified",
    "approved": "badge-verified",
    "blocked": "badge-blocked",
    "failed": "badge-blocked",
    "escalated": "badge-blocked",
    "awaiting_approval": "badge-neutral",
    "withdrawn": "badge-neutral",
}


@dataclass(frozen=True)
class TraceRow:
    """One event of the agent trace as the table draws it."""

    number: int
    clock: str  # the time of day in UTC, to the second
    iso: str  # the full timestamp in UTC, for `<time datetime>`
    agent: str
    task: str
    source: str
    result: str
    status: str
    status_style: str


def trace_rows(trace: Sequence[TraceEvent]) -> list[TraceRow]:
    """The trace in the order it was recorded, numbered from 1, with times in UTC.

    Every event is shown, with nothing left out or reworded: the trace is the evidence that each
    agent did only what it was allowed to, so the page must show what was recorded.
    """
    rows = []
    for number, event in enumerate(trace, start=1):
        utc = event.timestamp.astimezone(UTC)
        rows.append(
            TraceRow(
                number=number,
                clock=f"{utc:%H:%M:%S}",
                iso=utc.isoformat(),
                agent=AGENT_WORDS[event.agent],
                task=event.task,
                source=event.tool_or_source,
                result=event.result,
                status=STATUS_WORDS[event.status],
                status_style=_STATUS_STYLE[event.status],
            )
        )
    return rows


# --- the escalation screen --------------------------------------------------------------

# What the person is told, in plain words, for each step of the run that can end with staff. The
# keys are the task names the agents record (`TraceEvent.task`). A step not listed here still gets
# the last line, so a new escalation point can never leave the screen blank. A test checks that no
# two steps share a headline.
STEP_HEADLINES = {
    "extract goal": "We could not safely read your request.",
    "find nearest service": "We could not find a service that matches your request.",
    "decide eligibility": "We could not make an eligibility decision.",
    "plan journey": "We could not put a travel plan together.",
    "verify plan": "We could not check the plan against its sources.",
    "re-plan": "We could not find a plan that works, and there is nothing more we may try.",
    "request simulated booking": "A part of the booking you chose was no longer available.",
    "release data": "We could not finish sharing the details you approved.",
    "approve data release": "We could not prepare your sharing choices.",
}
UNKNOWN_STEP_HEADLINE = "We could not finish this request."

FALLBACK_WORDS = {
    "initial": "The dates you asked for",
    "next_window": "The next appointment window",
    "telehealth": "A telehealth appointment",
}
ATTEMPT_WORDS = {
    "no_option": "nothing fitted",
    "blocked": "the plan could not be verified",
    "rejected": "you sent the plan back",
}


def _sentence(text: str) -> str:
    """A recorded reason written as a sentence: capital first letter, full stop at the end."""
    text = text.strip()
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def escalation_context(state: GraphState) -> dict[str, Any]:
    """What the escalation screen says, worked out from the run's own record and nothing else.

    The reason is the one the run recorded just before it handed over to staff (the last "failed"
    event before the last "escalated" one), shown as recorded. What was and was not done is read
    from the state, so the screen can never say "nothing was booked" about a run that booked.
    """
    trace = list(state.trace)
    handover = max((i for i, e in enumerate(trace) if e.status == "escalated"), default=None)
    failed = None
    if handover is not None:
        failed = next((e for e in reversed(trace[:handover]) if e.status == "failed"), None)

    requested = [b for b in state.bookings if b.status == "requested"]
    done = [
        "Nothing was booked."
        if not requested
        else "These SIMULATED requests had already been made: "
        + ", ".join(f"{b.kind} {b.reference}" for b in requested)
        + ".",
        "No personal details were shared."
        if not state.releases
        else "Details had already been shared with: "
        + ", ".join(r.release.recipient for r in state.releases)
        + ".",
    ]
    tried = [
        f"{FALLBACK_WORDS[a.fallback]}, {format_date(a.window.earliest_date)} to "
        f"{format_date(a.window.latest_date)}: {ATTEMPT_WORDS[a.why]}"
        for a in state.attempts
    ]
    return {
        "headline": STEP_HEADLINES.get(failed.task, UNKNOWN_STEP_HEADLINE)
        if failed
        else UNKNOWN_STEP_HEADLINE,
        "reason": _sentence(failed.result) if failed else None,
        "done": done,
        "tried": tried,
        "own_words": state.goal_text,
        "has_plan": bool(state.plan),
    }
