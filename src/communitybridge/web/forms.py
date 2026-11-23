"""Reading and checking what the gate forms send back.

The browser sends text; each gate's answer is a typed model. This module is the strict step between
them, so a malformed or hostile POST is turned into a message the
person can act on, and never reaches the graph (a bad answer leaves the gate closed).

Design points:
- **A field that is not sent leaves that fact as it was.** An unchecked checkbox sends nothing, so
  it cannot tell "no" from "not asked", and a script that posts only a decision would flip every
  fact it did not mention. Yes/No radio buttons are used, and a missing one means "unchanged".
- **Only the facts the page offers can change** (`EDITABLE_FACTS`). Any other field in the POST is
  ignored, so a form cannot set a clinician's authorisation or anything resembling a decision.
- **Messages name the question, never echo what was typed.**
"""

import re
from collections.abc import Mapping
from datetime import date

from pydantic import ValidationError

from communitybridge.models.facts import ConfirmedFacts
from communitybridge.web.present import AGE_RANGE, EDITABLE_FACTS

GOAL_MAX_CHARS = 2000  # a request is a sentence or two; the cap stops a page-sized paste
DECISIONS = ("confirm", "edit", "decline")


def check_goal_text(raw: str) -> tuple[str, str | None]:
    """The request trimmed, and a problem message if it cannot be used."""
    text = raw.strip()
    if not text:
        return text, "Please type your request in the box."
    if len(text) > GOAL_MAX_CHARS:
        return text, (
            f"Your request is too long ({len(text)} characters). "
            f"Please keep it under {GOAL_MAX_CHARS} characters."
        )
    return text, None


def parse_fact_answers(
    current: ConfirmedFacts, submitted: Mapping[str, str | None]
) -> tuple[ConfirmedFacts | None, list[str]]:
    """The facts with the person's corrections applied, or the problems found.

    Returns `(None, [])` when the form sent none of the editable fields, meaning "no change".
    """
    updates: dict[str, object] = {}
    problems: list[str] = []
    for name, spec in EDITABLE_FACTS.items():
        raw = submitted.get(name)
        if raw is None:
            continue
        if spec.kind == "yes_no":
            if raw not in ("yes", "no"):
                problems.append(f"Please answer Yes or No to: {spec.question}")
            else:
                updates[name] = raw == "yes"
        else:
            low, high = AGE_RANGE
            # isdecimal() accepts only 0-9, so "4.5", "-1", "1e2" and spaces are refused here.
            if not raw.isdecimal() or not low <= int(raw) <= high:
                problems.append(f"{spec.question} Type a whole number from {low} to {high}.")
            else:
                updates[name] = int(raw)
    if problems or not updates:
        return None, problems
    try:
        return ConfirmedFacts.model_validate({**current.model_dump(), **updates}), []
    except ValidationError:
        # The checks above should make this unreachable; refuse rather than guess if it is not.
        return None, ["Some of your answers could not be used. Please check them and try again."]


# --- Gates 2, 3 and 4 -----------------------------------------------------------------

PLAN_DECISIONS = ("approve", "reject")
MAX_NAME_CHARS = 100
EARLIEST_BIRTH_YEAR = 1900
# Digits, spaces and the few marks a phone number carries; 6 to 20 characters long.
_PHONE = re.compile(r"[0-9+()\- ]{6,20}")


def parse_plan_decision(fields: Mapping[str, str]) -> tuple[dict[str, str] | None, list[str]]:
    """Gate 2: approve the plan, or send it back. Anything else leaves the gate closed."""
    decision = fields.get("decision", "")
    if decision not in PLAN_DECISIONS:
        return None, ["Please choose one of the two buttons: approve the plan, or send it back."]
    return {"decision": decision}, []


def _contact(fields: Mapping[str, str], today: date) -> tuple[dict[str, object] | None, list[str]]:
    """Gate 3's contact questions, as the dict `ClientContact` is built from.

    `synthetic` is true only because the person ticked the box saying the details are made up (hard
    rule 1). The server cannot tell a made-up name from a real one, so the page says plainly that
    this prototype is for made-up details, and the box is how that is put to them.
    """
    problems: list[str] = []
    names = {}
    for field, question in (("given_name", "first name"), ("family_name", "last name")):
        value = fields.get(field, "").strip()
        if not value:
            problems.append(f"Please type your {question}.")
        elif len(value) > MAX_NAME_CHARS:
            problems.append(f"Your {question} is too long (the limit is {MAX_NAME_CHARS}).")
        names[field] = value
    phone = fields.get("contact_phone", "").strip()
    if not _PHONE.fullmatch(phone):
        problems.append(
            "Please type a phone number: 6 to 20 characters, using digits, spaces and + ( ) - only."
        )
    born_text = fields.get("date_of_birth", "").strip()
    born: date | None = None
    try:
        born = date.fromisoformat(born_text)
    except ValueError:
        problems.append("Please give your date of birth as a real date (year-month-day).")
    if born is not None and not date(EARLIEST_BIRTH_YEAR, 1, 1) <= born <= today:
        problems.append(
            f"Your date of birth must be a real date in the past, after {EARLIEST_BIRTH_YEAR}."
        )
        born = None
    mobility = fields.get("mobility_assistance_needed", "")
    if mobility not in ("yes", "no"):
        problems.append("Please answer Yes or No: do you need help boarding or with seating?")
    if fields.get("made_up") != "yes":
        problems.append(
            "Please tick the box to say these details are made up. This prototype is for "
            "made-up details only."
        )
    if problems or born is None:
        return None, problems
    return {
        "synthetic": True,
        **names,
        "contact_phone": phone,
        "date_of_birth": born.isoformat(),
        "mobility_assistance_needed": mobility == "yes",
    }, []


def parse_booking_answer(
    fields: Mapping[str, str], option_count: int, today: date
) -> tuple[dict[str, object] | None, list[str]]:
    """Gate 3: pick an option and give contact details, or decline.

    A decline takes nothing else from the form, so contact details typed before changing one's mind
    are never kept.
    """
    decision = fields.get("decision", "")
    if decision == "decline":
        return {"decision": "decline"}, []
    if decision != "approve":
        return None, ["Please choose one of the two buttons: ask for the booking, or decline."]
    problems: list[str] = []
    option = fields.get("option", "")
    chosen = int(option) if option.isdecimal() and int(option) < option_count else None
    if chosen is None:
        problems.append("Please choose one of the options offered.")
    contact, contact_problems = _contact(fields, today)
    problems.extend(contact_problems)
    if problems or contact is None or chosen is None:
        return None, problems
    return {"decision": "approve", "option": chosen, "contact": contact}, []


def parse_consent_answer(
    fields: Mapping[str, str], offered: Mapping[str, str]
) -> tuple[dict[str, object] | None, list[str]]:
    """Gate 4: one decision for each action offered, and nothing for any other.

    `offered` maps each action to the words that describe it, used in the message. There is no
    default and no "approve all": an action with no clear answer leaves the whole gate closed.
    """
    decided: dict[str, str] = {}
    problems: list[str] = []
    for action, words in offered.items():
        answer = fields.get(f"decision-{action}", "")
        if answer in ("approve", "decline"):
            decided[action] = answer
        else:
            problems.append(f"Please decide each action. Not yet decided: {words}")
    if problems:
        return None, problems
    return {"decisions": decided}, []
