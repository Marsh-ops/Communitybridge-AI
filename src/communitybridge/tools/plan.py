"""Plan builder and checker: the statements a plan makes, and the Verifier's verdict on them.

The Verifier Agent's one tool. It does two things, both without a model:

1. `build_claims` turns what the other agents produced (the service found, the rules engine's
   result and its reasons, the logistics result and the options) into the *statements* a plan
   makes, each with the sources it relies on. A language model's wording of the decision is not an
   input: it is shown apart from the plan, labelled, and has no parameter here.
   The sources are taken from the tool results, never written in by hand: a rule hit brings its
   PATS section, the service brings its record, an option brings the route, stay and sections it
   was built from. So a statement has a citation because the thing it says came from somewhere,
   not because someone remembered to add one.
2. `verify_plan` hands those claims to the Verifier tool, which blocks any statement with
   no source, an unknown PATS section, a stale source or a conflicting claim.

Statements about the client's eligibility or their journey are *core*: if one is blocked the plan
cannot be shown with a hole in it and goes back to be re-planned (the design diagram). The others (the
engine's reasons, notices, entitlements, modes PATS does not subsidise) stay in the plan, shown as
blocked with the reason.

Questions for the client are not statements and are not checked here: `questions_for_user` collects
the ones the other tools raised (an entitlement that depends on an unanswered question, a mode
PATS does not name), so they can be put to the user at the plan gate.

After a re-plan the plan says why it differs from what the user asked for: a note that it
uses the next appointment window and which dates that is, or, for the telehealth attempt, a note
that it is the specialist's own telehealth appointment, that there is no travel or stay so no PATS
travel or accommodation subsidy is needed (§6.5), and that the eligible in-person appointment is
still open. A telehealth plan makes no statement about a journey it does not involve (the
travel mode, modes PATS does not subsidise, travel entitlements), and asks the user nothing about
one.

Every figure in a statement is read from a tool result and formatted, so a plan can never say a
number the tools did not work out. Money is in Australian dollars from the SIMULATED fare sheet and
the rule rates, and the statements say so.
"""

from collections.abc import Iterable
from datetime import date

from communitybridge.models import EligibilityExplanation, Service
from communitybridge.models.common import Citation
from communitybridge.models.engine import EligibilityOutcome, EligibilityResult
from communitybridge.models.logistics import LogisticsAssessment
from communitybridge.models.plan import LogisticsOption
from communitybridge.models.replan import DateWindow, Fallback
from communitybridge.models.verification import Claim, PlanCheck
from communitybridge.simulated.common import weekday_full
from communitybridge.tools.rules_engine import EngineContext
from communitybridge.tools.verifier import (
    catalogue_from_context,
    citation_for_section,
    verify_claims,
)

OUTCOME_WORDS: dict[EligibilityOutcome, str] = {
    "eligible": "eligible",
    "not_eligible": "not eligible",
    "needs_review": "needs review by a PATS authorised officer",
}

# The nearest-specialist rule, cited for the service statement (PATS §6.4.6).
NEAREST_SECTION = "6.4.6"
# Travel by a mode PATS does not subsidise (PATS §6.5.8).
NON_ELIGIBLE_TRAVEL_SECTION = "6.5.8"
# The travel section: a telehealth plan involves no travel, so no travel subsidy is needed (§6.5).
TELEHEALTH_SECTION = "6.5"


def _section_key(section: str) -> tuple[int, ...]:
    return tuple(int(part) for part in section.split("."))


def _pats(sections: Iterable[str], context: EngineContext) -> list[Citation]:
    """PATS citations for the sections, each once, in numeric order."""
    unique = sorted(set(sections), key=_section_key)
    return [citation_for_section(section, context.rules) for section in unique]


def _words(value: str) -> str:
    return value.replace("_", " ")


def _money(amount: float) -> str:
    return f"${amount:,.2f}"


def option_statement(number: int, option: LogisticsOption) -> str:
    """One option as a sentence, with only figures the tools worked out."""
    start = option.appointment_start
    text = f"Option {number}: appointment {start:%Y-%m-%d %H:%M} ({weekday_full(start.date())}); "
    if option.outbound_date is None:
        text += "a telehealth appointment, so no travel or stay"
    else:
        text += (
            f"travel by {_words(option.travel_mode)} out on {option.outbound_date} and back on "
            f"{option.return_date}, {option.nights} night(s)"
        )
        if option.accommodation_name:
            text += f" at {option.accommodation_name}"
        text += f", for {option.travellers} traveller(s)"
        if option.travel_subsidy_aud is not None:
            text += f"; PATS would pay up to {_money(option.travel_subsidy_aud)} towards the travel"
        if option.accommodation_subsidy_aud is not None:
            text += f" and up to {_money(option.accommodation_subsidy_aud)} towards the stay"
    return text + " (SIMULATED availability and fares)."


def attempt_claims(
    service: Service, fallback: Fallback, window: DateWindow | None, context: EngineContext
) -> list[Claim]:
    """The note a re-planned plan carries about why it differs from what the user asked for."""
    record = Citation(
        source=service.source,
        section_or_record=f"service {service.service_id}",
        last_verified=service.last_verified,
    )
    if fallback == "next_window":
        if window is None:
            raise ValueError("the dates of the next window are needed to describe it")
        return [
            Claim(
                statement=(
                    "None of the dates you asked for had an arrangement that works, so this plan "
                    f"uses the next appointment window: {window.earliest_date} to "
                    f"{window.latest_date}."
                ),
                citations=[record],
            )
        ]
    if fallback == "telehealth":
        return [
            Claim(
                statement=(
                    "No in-person arrangement could be made, so this plan is a telehealth "
                    "appointment with the specialist, which needs no travel or stay. No PATS "
                    "travel or accommodation subsidy is needed (§6.5). An eligible in-person "
                    "appointment is still open if you would rather wait."
                ),
                citations=[*_pats([TELEHEALTH_SECTION], context), record],
            )
        ]
    return []


def build_claims(
    service: Service,
    eligibility: EligibilityResult,
    reasons: EligibilityExplanation | None,
    logistics: LogisticsAssessment | None,
    options: list[LogisticsOption],
    context: EngineContext,
    fallback: Fallback = "initial",
    window: DateWindow | None = None,
) -> list[Claim]:
    """Every statement the plan makes, with the sources each came from. Read-only.

    `reasons` is the rules engine's own wording of its rule results, built by the rules. It is not
    the language model's explanation, which is never a plan statement.
    """
    journey = fallback != "telehealth"  # a telehealth plan involves no travel or stay
    decision = eligibility.decision
    town = service.town.replace("_", " ").title()
    claims = [
        Claim(
            statement=(
                f"The nearest {service.specialty} service is {service.name} "
                f"({service.mode}) in {town}."
            ),
            citations=[
                Citation(
                    source=service.source,
                    section_or_record=f"service {service.service_id}",
                    last_verified=service.last_verified,
                ),
                *_pats([NEAREST_SECTION], context),
            ],
            subject="nearest service",
            value=service.service_id,
            core=True,
        ),
        Claim(
            statement=f"Outcome of the PATS rules check: {OUTCOME_WORDS[eligibility.outcome]}.",
            citations=_pats((hit.section for hit in decision.rule_hits), context),
            subject="eligibility outcome",
            value=eligibility.outcome,
            core=True,
        ),
    ]
    if reasons is not None:
        claims.append(
            Claim(
                statement=reasons.text,
                citations=_pats(reasons.cited_sections, context),
            )
        )
    for notice in eligibility.notices:
        claims.append(
            Claim(
                statement=f"Not assessed (§{notice.section} {notice.title}): {notice.message}",
                citations=_pats([notice.section], context),
            )
        )
    claims += attempt_claims(service, fallback, window, context)
    if journey and eligibility.entitlements is not None:
        for hit in eligibility.entitlements.rule_hits:
            if hit.outcome == "pass":
                claims.append(Claim(statement=hit.reason, citations=_pats([hit.section], context)))

    if journey and logistics is not None:
        modes = logistics.modes
        if modes.chosen_mode is not None:
            spend = (
                f", up to {_money(modes.travel_subsidy_aud)}"
                if modes.travel_subsidy_aud is not None
                else ""
            )
            claims.append(
                Claim(
                    statement=(
                        "The most economical way of travelling that PATS pays for is "
                        f"{_words(modes.chosen_mode)}{spend} (SIMULATED fares)."
                    ),
                    citations=_pats((hit.section for hit in modes.rule_hits), context),
                    subject="travel mode",
                    value=modes.chosen_mode,
                    core=True,
                )
            )
        for mode in logistics.ineligible_modes_flagged:
            claims.append(
                Claim(
                    statement=f"PATS does not subsidise travel by {mode}.",
                    citations=_pats([NON_ELIGIBLE_TRAVEL_SECTION], context),
                )
            )
    for number, option in enumerate(options, start=1):
        claims.append(
            Claim(
                statement=option_statement(number, option),
                citations=list(option.citations),
                core=True,
            )
        )
    return claims


def questions_for_user(
    eligibility: EligibilityResult, logistics: LogisticsAssessment | None
) -> list[str]:
    """The questions the tools raised that the user has to answer, once each.

    They are for the plan gate: an answer can change an entitlement or the travel mode. They are
    not statements, so they are not checked for sources.
    """
    questions: list[str] = []
    if eligibility.entitlements is not None:
        questions += eligibility.entitlements.questions
    if logistics is not None:
        questions += logistics.questions
        questions += [
            f"PATS does not say whether travel by {mode} is subsidised. Please ask a PATS "
            "authorised officer."
            for mode in logistics.unrecognised_modes
        ]
    return list(dict.fromkeys(questions))


def verify_plan(
    service: Service,
    eligibility: EligibilityResult,
    reasons: EligibilityExplanation | None,
    logistics: LogisticsAssessment | None,
    options: list[LogisticsOption],
    context: EngineContext,
    today: date,
    fallback: Fallback = "initial",
    window: DateWindow | None = None,
) -> PlanCheck:
    """Build the plan's statements and check each one: verified, or blocked with the reason.

    The sources the Verifier accepts are the PATS sections in the rules data and the records in the
    dataset (services, towns) plus the simulated routes and accommodation the options use. `today`
    is passed in so a check is repeatable. The result also carries the questions to put to the
    user: what the verdict asks them to confirm, and what the other tools asked.
    """
    claims = build_claims(
        service, eligibility, reasons, logistics, options, context, fallback, window
    )
    record_ids = {
        record
        for option in options
        for record in (option.outbound_route_id, option.return_route_id, option.accommodation_id)
        if record is not None
    }
    catalogue = catalogue_from_context(context, extra_record_ids=record_ids)
    report = verify_claims(claims, catalogue, today=today)
    # A telehealth plan has no journey, so the questions about one do not apply to it.
    asked = questions_for_user(eligibility, logistics) if fallback != "telehealth" else []
    questions = [*asked, *report.escalations]
    return PlanCheck(report=report, questions=list(dict.fromkeys(questions)))
