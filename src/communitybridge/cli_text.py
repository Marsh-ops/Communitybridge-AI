"""The text the terminal interface shows, and the way it reads typed answers.

Serves every agent's gate: Gate 1 (Coordinator: confirm the details), Gate 2 (Coordinator:
approve the plan), Gate 3 (Logistics: choose a SIMULATED booking request) and Gate 4 (Consent:
approve, action by action, the data each request needs). It is the terminal twin of the web screens
built later, and it lives apart from `cli.py` so every sentence a person reads can be tested
without running a graph.

Why it is built this way:
- Everything here is a pure function from data to a string. Nothing reads the keyboard or calls a
  tool, so it cannot approve anything.
- A plan statement is shown with its source, or shown BLOCKED with the reason.
- Anything simulated carries the SIMULATED badge.
- Gate 4 lists the exact values that would be shared, and shows a missing value as MISSING rather
  than as a blank, so nobody approves something they cannot see.
- A typed answer is read against a short list of accepted words. An empty or unclear line is *not*
  an approval: `choose_from` returns None and the caller asks again.
"""

import textwrap
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from communitybridge.graph.state import GraphState
from communitybridge.models import PlanItem, TraceEvent
from communitybridge.models.common import Citation
from communitybridge.models.consent_release import ReleaseRecord, WithdrawalNotice
from communitybridge.models.eligibility import MODEL_WORDING_LABEL
from communitybridge.models.simulated import BookingRequest

BADGE = "[SIMULATED]"
WIDTH = 100  # statements are wrapped to this many columns so the plan reads on a normal terminal

# --- typed answers -----------------------------------------------------------------------------

# Each map is every word a person may type at that question, in lower case. There is deliberately no
# "yes", "ok" or empty-line default: an approval has to be a clear word, not a slip of the keyboard.
DETAILS_CHOICES = {
    "c": "confirm",
    "confirm": "confirm",
    "e": "edit",
    "edit": "edit",
    "d": "decline",
    "decline": "decline",
}
PLAN_CHOICES = {"a": "approve", "approve": "approve", "r": "reject", "reject": "reject"}
ACTION_CHOICES = {"a": "approve", "approve": "approve", "d": "decline", "decline": "decline"}


def choose_from(choices: Mapping[str, str], line: str) -> str | None:
    """The decision a typed line stands for, or None if it is not one of the accepted words."""
    return choices.get(line.strip().lower())


def gate_three_choice(line: str, option_count: int) -> dict[str, Any] | None:
    """Gate 3 is answered with an option's number (counting from 1 on screen) or "d" to decline.

    The graph counts options from 0, so the number is converted here and nowhere else. Anything that
    is not a whole number between 1 and `option_count` is not an answer.
    """
    text = line.strip().lower()
    if text in ("d", "decline"):
        return {"decision": "decline"}
    if text.isascii() and text.isdigit() and 1 <= int(text) <= option_count:
        return {"decision": "approve", "option": int(text) - 1}
    return None


# --- the plan and the trace --------------------------------------------------------------------


def source_lines(citations: Sequence[Citation], indent: str) -> list[str]:
    """One line per source and verification date, listing every section or record cited from it.

    A plan statement can rest on a dozen PATS sections; a line each would bury the statement. The
    source name, the date it was last verified and every section are all still shown, so the
    Verifier's freshness check can be seen on screen.
    """
    grouped: dict[tuple[str, date], list[str]] = {}
    for citation in citations:
        parts = grouped.setdefault((citation.source, citation.last_verified), [])
        if citation.section_or_record not in parts:
            parts.append(citation.section_or_record)
    return [
        f"{indent}source: {source} (last verified {verified.isoformat()}): {', '.join(parts)}"
        for (source, verified), parts in grouped.items()
    ]


def render_plan(plan: Sequence[PlanItem], questions: Sequence[str]) -> str:
    """Every plan statement with its sources, or marked BLOCKED with the reason."""
    if not plan:
        return "THE PLAN\n  The plan has no statements.\n"
    verified = sum(1 for item in plan if item.status == "verified")
    blocked = len(plan) - verified
    lines = [f"THE PLAN  ({verified} verified, {blocked} blocked)"]
    for number, item in enumerate(plan, start=1):
        tag = "VERIFIED" if item.status == "verified" else "BLOCKED"
        lines.append(
            textwrap.fill(
                f"{number:>2}. [{tag}] {item.statement}",
                width=WIDTH,
                initial_indent="  ",
                subsequent_indent=" " * 11,
            )
        )
        lines.extend(source_lines(item.citations, " " * 8))
        if item.status == "blocked":
            lines.append(f"        why it is blocked: {item.blocked_reason}")
    if questions:
        lines.append("Questions for you:")
        lines.extend(f"  - {question}" for question in questions)
    return "\n".join(lines) + "\n"


def render_model_wording(label: str, text: str, cited_sections: Sequence[str]) -> str:
    """A language model's wording of the decision, under its label and apart from the plan.

    It is not one of the plan's statements: it is not numbered, counted or tagged VERIFIED. The
    label says what its check covered (the sections it cites and the outcome it states) and that it
    is not verified.
    """
    lines = [
        "IN PLAIN WORDS  (a language model's wording; not one of the plan's statements)",
        textwrap.fill(label, width=WIDTH, initial_indent="  ", subsequent_indent="  "),
        textwrap.fill(text, width=WIDTH, initial_indent="  > ", subsequent_indent="    "),
    ]
    if cited_sections:
        lines.append("  it cites: " + ", ".join(f"§{section}" for section in cited_sections))
    return "\n".join(lines) + "\n"


def render_trace(trace: Sequence[TraceEvent]) -> str:
    """The agent trace the project brief asks for: task, tool or source, result and status."""
    if not trace:
        return "AGENT TRACE\n  There are no events in the trace.\n"
    lines = [f"AGENT TRACE  ({len(trace)} events, times in UTC)"]
    for number, event in enumerate(trace, start=1):
        stamp = event.timestamp.strftime("%H:%M:%S")
        lines.append(f"{number:>3}  {stamp}  {event.agent:<17} {event.task}   [{event.status}]")
        lines.append(f"       tool/source: {event.tool_or_source}")
        lines.append(f"       result: {event.result}")
    return "\n".join(lines) + "\n"


# --- the four gates ----------------------------------------------------------------------------


class UnknownGateError(ValueError):
    """The graph asked a question this interface has no screen for."""


def _value(value: object) -> str:
    if value is None:
        return "not stated"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "none"
    return str(value)


def _cost(total: float | None) -> str:
    """A request's cost. An appointment request has none, which is not the same as 'unknown'."""
    return "no charge shown" if total is None else f"${total:,.2f}"


def _when(iso: str) -> str:
    """An ISO date and time as the person reads it: 2026-11-10T09:30:00 becomes 2026-11-10 09:30."""
    return iso.replace("T", " ")[:16]


def _money(amount: float | None) -> str:
    return "not worked out" if amount is None else f"${amount:,.2f}"


def _gate_one(payload: Mapping[str, Any]) -> list[str]:
    lines = ["GATE 1 of 4: is this what you meant?", "What I understood from your request:"]
    for name, value in payload["goal"].items():
        if name != "confirmed_by_user":
            lines.append(f"  {name}: {_value(value)}")
    lines.append("Other details the rules will use (given, or assumed for a typical adult):")
    for name, value in payload["facts"].items():
        if name == "indigenous_status" and value is None:
            # None means "use what the request says" (models/facts.py), not "unknown".
            lines.append(f"  {name}: taken from your request above")
        else:
            lines.append(f"  {name}: {_value(value)}")
    return lines


def _gate_two(payload: Mapping[str, Any]) -> list[str]:
    lines = ["GATE 2 of 4: do you approve this plan?"]
    attempt = payload.get("fallback", "initial")
    if attempt != "initial":
        words = {"next_window": "the next appointment window", "telehealth": "telehealth"}
        lines.append(f"This is a re-plan: it uses {words.get(attempt, attempt)}.")
    lines.append(render_plan([PlanItem.model_validate(i) for i in payload["plan"]], []).rstrip())
    wording = payload.get("model_wording")
    if wording:
        lines.append(
            render_model_wording(
                wording["label"], wording["text"], wording["cited_sections"]
            ).rstrip()
        )
    questions = payload.get("questions", [])
    if questions:
        lines.append("Questions for you:")
        lines.extend(f"  - {question}" for question in questions)
    lines.append(
        "Approving does not book anything; you choose a booking next."
        if payload.get("booking_follows")
        else "This plan is information only, so approving it ends the run."
    )
    return lines


def _option(number: int, option: Mapping[str, Any]) -> list[str]:
    lines = [
        f"Option {number}  {BADGE}",
        f"  travel: {option['travel_mode']} for {option['travellers']} traveller(s)",
        f"  go {_value(option.get('outbound_date'))}, return {_value(option.get('return_date'))}"
        f", {option.get('nights', 0)} night(s)"
        + (f" at {option['accommodation_name']}" if option.get("accommodation_name") else ""),
        f"  appointment: {_when(option['appointment_start'])}",
        f"  PATS would pay up to: travel {_money(option.get('travel_subsidy_aud'))}, "
        f"accommodation {_money(option.get('accommodation_subsidy_aud'))}",
    ]
    lines.extend(f"  note: {note}" for note in option.get("notes", []))
    lines.extend(source_lines([Citation.model_validate(c) for c in option["citations"]], "  "))
    return lines


def _gate_three(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        f"GATE 3 of 4: choose a booking request {BADGE}",
        "Nothing real is booked: these are requests to SIMULATED services.",
    ]
    for number, option in enumerate(payload["options"], start=1):
        lines.extend(_option(number, option))
    return lines


def _action(number: int, count: int, prompt: Mapping[str, Any]) -> list[str]:
    lines = [
        f"Action {number} of {count}: {prompt['action']}  {BADGE}",
        f"  what: {prompt['description']}",
        f"  goes to: {prompt['recipient']} ({prompt['recipient_type']})",
        f"  why: {prompt['purpose']}",
        "  shares only:",
    ]
    for field in prompt["fields"]:
        if field["name"] in prompt.get("missing", []) or field.get("value") is None:
            shown = "MISSING (this must be supplied before the action can be approved)"
        else:
            shown = str(field["value"])
        lines.append(f"    - {field['name']} ({field['description']}): {shown}")
    lines.append(f"  if you withdraw later: {prompt['on_withdrawal']}")
    return lines


def _gate_four(payload: Mapping[str, Any]) -> list[str]:
    actions = payload["actions"]
    lines = [
        f"GATE 4 of 4: what may be shared, action by action {BADGE}",
        "Each action is decided on its own. Declining one does not affect the others.",
    ]
    for number, prompt in enumerate(actions, start=1):
        lines.extend(_action(number, len(actions), prompt))
    return lines


_GATES = {
    "confirm_details": _gate_one,
    "approve_plan": _gate_two,
    "approve_booking": _gate_three,
    "approve_release": _gate_four,
}


def render_gate(payload: Mapping[str, Any]) -> str:
    """The screen for one gate: what the user is being asked to decide, in full.

    `payload` is what the graph's `interrupt()` handed out, so this shows exactly what the gate
    holds and nothing else. If the last answer was not understood the reason comes first.
    """
    gate = payload.get("gate")
    build = _GATES.get(str(gate))
    if build is None:
        raise UnknownGateError(f"no screen for the gate {gate!r}")
    lines = []
    if "error" in payload:
        lines.append(f"Your last answer was not understood: {payload['error']}")
    lines.extend(build(payload))
    return "\n".join(lines) + "\n"


# --- what happened -----------------------------------------------------------------------------


def render_actions(
    bookings: Sequence[BookingRequest],
    releases: Sequence[ReleaseRecord],
    withdrawals: Sequence[WithdrawalNotice],
) -> str:
    """The SIMULATED requests made and data released, or a plain statement that there were none."""
    lines = ["SIMULATED ACTIONS  (nothing real was booked, paid for or sent)"]
    if bookings:
        lines.append("  Booking requests:")
        for booking in bookings:
            lines.append(
                f"    {booking.reference}  {booking.kind}  {booking.status}  "
                f"{_cost(booking.total_aud)}  {booking.description}  {BADGE}"
            )
    else:
        lines.append("  No booking request was made.")
    if releases:
        lines.append("  Data released:")
        for record in releases:
            release = record.release
            shared = ", ".join(f"{name}={value}" for name, value in release.data.items())
            lines.append(
                f"    {record.reference}  {release.action} to {release.recipient}: "
                f"{shared}  {BADGE}"
            )
    else:
        lines.append("  No data was released.")
    for notice in withdrawals:
        lines.append(f"  Consent withdrawn for {notice.action}: {notice.message}")
    return "\n".join(lines) + "\n"


_STATUS_WORDS = {
    "planned": "planned: the plan was approved and any approved requests were made",
    "declined": "declined: you said no, so no action was taken",
    "escalated": "escalated: a staff member needs to take this request",
    "in_progress": "stopped before it finished",
    "awaiting_approval": "stopped, waiting for your answer",
}


def render_outcome(state: GraphState) -> str:
    """How the run ended: the status, what the rules engine decided, and how many re-plans."""
    lines = ["RESULT", f"  status: {_STATUS_WORDS.get(state.status, state.status)}"]
    if state.service is not None:
        lines.append(f"  service: {state.service.name} ({state.service.town})")
    if state.eligibility is not None:
        decision = state.eligibility.decision
        lines.append(
            f"  rules engine: {state.eligibility.outcome} "
            f"(PATS rules version {decision.rules_version}; decided by the rules engine, "
            "not by a language model)"
        )
    if state.explanation is not None:
        # The model's wording, labelled: the same block Gate 2 shows.
        wording = render_model_wording(
            MODEL_WORDING_LABEL, state.explanation.text, state.explanation.cited_sections
        )
        lines.extend(wording.rstrip().splitlines())
    lines.append(f"  re-plans used: {state.replan_count} of 2")
    return "\n".join(lines) + "\n"
