"""The rules engine: `decide()` combines every PATS rule into one eligibility answer.

This is the deterministic rules engine that decides PATS eligibility (the core rule: no language
model can set or change an eligibility result). The Eligibility Agent only receives the result
and explains it. The module has two layers:

- The yes/no *gates*, which come first and are described below.
- `decide`, which runs the gates, the distance rules (`tools/distance.py`), the
  nearest-service rules (`tools/nearest_service.py`) and, for an eligible client, the
  entitlement rules (`tools/entitlements.py`), then combines them. Any rule that fails makes
  the client not eligible; otherwise any open review makes it "needs review"; otherwise the
  client is eligible. A failure outranks a review because a client who fails a rule is not
  eligible whatever a review later finds.

Each rule returns `RuleHit`s that cite its own id and PDF section, so every statement in the final
plan can be traced to the guideline. The gates are:

- `check_residency`: §6.1 (holidays and visiting are excluded; itinerant workers are reviewed);
- `check_apy_lands`: §6.1.1 (Aboriginal APY Lands residents are covered by Nganampa Health);
- `check_specialist`: §9.2 accepted specialities, §6.4.1 recognised specialists and
  sub-specialties, §6.4.2 approved services;
- `check_professional`: §6.4.4 (allied health, nursing, midwifery and GPs are excluded);
- `check_appointment`: §6.4.5 (second opinions, cosmetic-only surgery, monitoring programs and
  trials other than phase IV are excluded).

Design choices (the project plan stores the rules as versioned data):
- Every list comes from data/rules/pats_v26_4.yaml. The code fixes only how a client's situation
  maps onto a list entry (for example `on_holidays` finds the entry that mentions "holiday").
- If an entry the code needs is missing from the data, or the data says something the code does
  not implement, the gate raises `RuleParameterError`. A missing entry must never mean "no
  problem found", because that would let an excluded client through.
- Where the PDF is discretionary or a fact is unknown, the answer is `needs_review` (an
  authorised officer, §8.3), not a guess.

Not modelled, on purpose: the dental exceptions in §6.4.3, the organ and tissue donor exemption
in §6.1, and Medicare enrolment and other-provider assistance (§6.2, §6.3), which are assumed
preconditions.
"""

import re
from dataclasses import dataclass

from communitybridge.db.seed import (
    load_non_specialist_categories,
    load_services,
    load_towns,
)
from communitybridge.models.dataset import ServiceRecord, Town
from communitybridge.models.distance import TravelDistanceInput
from communitybridge.models.eligibility import Decision, RuleHit
from communitybridge.models.engine import EligibilityOutcome, EligibilityResult
from communitybridge.models.entitlements import EntitlementInput
from communitybridge.models.facts import ConfirmedFacts
from communitybridge.models.gates import (
    AppointmentType,
    GateAssessment,
    GateFacts,
    ResidencySituation,
    status_for,
)
from communitybridge.models.nearest import NearestServiceInput
from communitybridge.models.request import GoalRequest
from communitybridge.models.rules import Rule, RuleSet
from communitybridge.tools.distance import (
    DistanceTable,
    assess_travel_distance,
    check_away_from_residence,
    load_distance_table,
)
from communitybridge.tools.entitlements import assess_entitlements
from communitybridge.tools.nearest_service import assess_nearest_service
from communitybridge.tools.rule_access import (
    RuleParameterError,
    entry_with,
    find_rule,
    make_hit,
    norm,
    param,
    require,
    text_list,
)
from communitybridge.tools.rules_loader import load_rules
from communitybridge.tools.service_query import ServiceQuery

RESIDENCY_RULE = "PATS-6.1-RESIDENCY"
APY_RULE = "PATS-6.1.1-APY"
SPECIALITIES_RULE = "PATS-9.2-SPECIALITIES"
RECOGNISED_RULE = "PATS-6.4.1-RECOGNISED"
SUBSPECIALTIES_RULE = "PATS-6.4.1-SUBSPECIALTIES"
APPROVED_RULE = "PATS-6.4.2-APPROVED-SERVICES"
PROFESSIONALS_RULE = "PATS-6.4.4-INELIGIBLE-PROFESSIONALS"
APPOINTMENTS_RULE = "PATS-6.4.5-INELIGIBLE-APPOINTMENTS"

# The word that finds the matching entry in the rules data for each situation or appointment type.
_RESIDENCY_KEYWORDS: dict[ResidencySituation, str] = {
    "on_holidays": "holiday",
    "visiting_family_and_friends": "visiting family",
    "itinerant_worker": "itinerant",
}
_APPOINTMENT_KEYWORDS: dict[AppointmentType, str] = {
    "second_opinion": "second opinion",
    "cosmetic": "cosmetic",
    "monitoring": "monitoring",
    "clinical_trial": "clinical trial",
    "clinical_trial_phase_iv": "clinical trial",
}
_MEDICAL_SPECIALIST = "medical specialist"


# --- §6.1 Residency --------------------------------------------------------------------------


def check_residency(facts: GateFacts, rules: RuleSet) -> RuleHit:
    """§6.1 Residency: the client must be a permanent South Australian resident.

    Returns `fail` for the situations the PDF lists as ineligible, `needs_review` for an itinerant
    worker (the PDF only says they "will be considered", so a person decides:), and
    `pass` otherwise. The situations come from the rules data, not from this code.
    """
    rule = find_rule(rules, RESIDENCY_RULE)
    require(rule, "requires_permanent_sa_residence", True)

    situation = facts.residency_situation
    if situation == "itinerant_worker":
        entry = entry_with(
            text_list(rule, "needs_review_situations"),
            _RESIDENCY_KEYWORDS[situation],
            rule,
            "needs_review_situations",
        )
        # The PDF says itinerant workers "will be considered", which is a discretion.
        return make_hit(
            rule,
            "needs_review",
            f"A {entry} is not automatically excluded, but the guideline only says they 'will be "
            "considered', so a PATS authorised officer must review it (§8.3).",
        )
    if situation is not None:
        entry = entry_with(
            text_list(rule, "ineligible_situations"),
            _RESIDENCY_KEYWORDS[situation],
            rule,
            "ineligible_situations",
        )
        return make_hit(rule, "fail", f"Clients who are {entry} are not eligible for subsidy.")

    if not facts.permanent_sa_resident:
        return make_hit(
            rule,
            "fail",
            "PATS subsidies need a permanent South Australian resident, and this client is not "
            "one (holiday accommodation is not a permanent residence).",
        )
    return make_hit(rule, "pass", "The client is a permanent South Australian resident.")


# --- §6.1.1 APY Lands residents ----------------------------------------------------------------


def check_apy_lands(facts: GateFacts, rules: RuleSet) -> list[RuleHit]:
    """No hit at all away from the APY Lands: the rule has nothing to say about those clients."""
    if not facts.lives_on_apy_lands:
        return []
    rule = find_rule(rules, APY_RULE)
    # An unstated identity is neither excluded nor cleared, so the data must say it is reviewed.
    require(rule, "not_stated_outcome", "needs_review")
    excluded = {norm(status) for status in text_list(rule, "excluded_indigenous_status")}

    status = facts.indigenous_status
    if norm(status) in excluded:
        return [
            make_hit(
                rule,
                "fail",
                "Aboriginal people living on the APY Lands are not eligible for PATS subsidies: "
                "Nganampa Health Council manages their travel and accommodation.",
            )
        ]
    if status == "not_stated":
        return [
            make_hit(
                rule,
                "needs_review",
                "The client lives on the APY Lands, where PATS treats Aboriginal people "
                "differently, but their Indigenous status has not been stated. Ask the client, "
                "or a PATS authorised officer must review it (§8.3).",
            )
        ]
    return [
        make_hit(
            rule,
            "pass",
            "The client lives on the APY Lands but is not Aboriginal, so §6.1.1 does not exclude "
            "them and they are assessed under the other criteria.",
        )
    ]


# --- §9.2, §6.4.1, §6.4.2 specialists and approved services ---------------------------------------


def _split_qualifier(entry: str) -> tuple[str, str | None]:
    """'radiology services (only when referred ...)' -> the name and the bracketed text."""
    match = re.fullmatch(r"(.*?)\s*\(([^()]*)\)\s*", entry)
    return (match.group(1), match.group(2)) if match else (entry, None)


def _subspecialties(rule: Rule) -> dict[str, set[str]]:
    table = param(rule, "accepted_subspecialties")
    if not isinstance(table, dict) or not all(
        isinstance(key, str) and isinstance(items, list) and all(isinstance(i, str) for i in items)
        for key, items in table.items()
    ):
        raise RuleParameterError(
            f"{rule.rule_id}: accepted_subspecialties must map a speciality to a list of text"
        )
    return {norm(key): {norm(item) for item in items} for key, items in table.items()}


def check_specialist(facts: GateFacts, rules: RuleSet) -> list[RuleHit]:
    """Is this a specialist service PATS recognises? §9.2, then §6.4.1, then §6.4.2.

    Tried in that order and stops at the first match: an accepted speciality (§9.2), an accepted
    sub-specialty (§6.4.1), or an approved service such as dialysis (§6.4.2, which can need a
    person's review). If none match, both rules fail. Returns one hit per rule it applied, so the
    explanation can cite exactly those sections.
    """
    specialities = find_rule(rules, SPECIALITIES_RULE)
    recognised = find_rule(rules, RECOGNISED_RULE)
    subspecialties = find_rule(rules, SUBSPECIALTIES_RULE)
    approved = find_rule(rules, APPROVED_RULE)
    sought = norm(facts.specialty)

    # §9.2 first: the list of accepted medical specialities. The engine cannot see the
    # practitioner's Medicare registration (§6.4.1), so an accepted speciality stands in for it.
    if sought in {norm(item) for item in text_list(specialities, "accepted_specialities")}:
        return [
            make_hit(
                specialities,
                "pass",
                f"{facts.specialty} is on the list of accepted medical specialities (§9.2).",
            ),
            make_hit(
                recognised,
                "pass",
                "The appointment is treated as being with a recognised medical specialist "
                f"because {facts.specialty} is an accepted speciality.",
            ),
        ]

    # §6.4.1 sub-specialties accepted "where applicable", for a parent that is not itself listed.
    if facts.subspecialty is not None:
        if norm(facts.subspecialty) in _subspecialties(subspecialties).get(sought, set()):
            return [
                make_hit(
                    subspecialties,
                    "pass",
                    f"{facts.subspecialty} is an accepted sub-specialty of {facts.specialty} "
                    "(§6.4.1).",
                ),
                make_hit(
                    recognised,
                    "pass",
                    "The appointment is treated as being with a recognised medical specialist "
                    "because the sub-specialty is accepted.",
                ),
            ]

    # §6.4.2: approved services such as chemotherapy and renal dialysis count in their own right.
    for entry in text_list(approved, "approved_services"):
        name, qualifier = _split_qualifier(entry)
        if sought not in {norm(entry), norm(name)}:
            continue
        if qualifier is not None and norm(qualifier).startswith("only when"):
            return [
                make_hit(
                    approved,
                    "needs_review",
                    f"{name} is an approved service ('{qualifier}'). That condition can't be "
                    "checked from the details given, so a PATS authorised officer must review "
                    "it (§8.3).",
                )
            ]
        return [
            make_hit(
                approved,
                "pass",
                f"{facts.specialty} is an approved medical specialist service (§6.4.2).",
            )
        ]

    return [
        make_hit(
            specialities,
            "fail",
            f"{facts.specialty} is not on the list of accepted medical specialities (§9.2).",
        ),
        make_hit(
            recognised,
            "fail",
            f"So the appointment is not with a recognised medical specialist (§6.4.1), and "
            f"{facts.specialty} is not an approved service (§6.4.2).",
        ),
    ]


# --- §6.4.4 Ineligible health professional services ----------------------------------------


def check_professional(facts: GateFacts, rules: RuleSet) -> RuleHit:
    """§6.4.4 Ineligible health professional services: the service must be a specialist's.

    A category that is neither "medical specialist" nor on the excluded list raises
    `RuleParameterError`, so a typo in the data stops the run instead of reading as "allowed".
    """
    rule = find_rule(rules, PROFESSIONALS_RULE)
    category = facts.professional_category
    if category is None or norm(category) == _MEDICAL_SPECIALIST:
        return make_hit(
            rule,
            "pass",
            "The service is provided by a medical specialist, not by one of the health "
            "professionals that §6.4.4 excludes.",
        )
    for entry in text_list(rule, "ineligible_professionals"):
        if norm(entry) == norm(category):
            return make_hit(
                rule,
                "fail",
                f"The service is provided by {entry}, who are not classified as medical "
                "specialists and do not meet PATS eligibility criteria.",
            )
    # A typo in the dataset must not read as "not excluded".
    raise RuleParameterError(
        f"{rule.rule_id}: professional category {category!r} is neither "
        f"{_MEDICAL_SPECIALIST!r} nor one of the excluded professionals in the rules data"
    )


# --- §6.4.5 Ineligible appointments ---------------------------------------------------------------


def check_appointment(facts: GateFacts, rules: RuleSet) -> RuleHit:
    """§6.4.5 Ineligible appointments: a standard appointment passes, a listed kind fails.

    Phase IV clinical trials are the one exception the PDF makes inside the list ("other than
    phase IV trials"), and the check refuses to run if the data no longer says so.
    """
    rule = find_rule(rules, APPOINTMENTS_RULE)
    entries = text_list(rule, "ineligible_situations")
    kind = facts.appointment_type
    if kind == "standard":
        return make_hit(
            rule,
            "pass",
            "The appointment is not one of the situations that do not meet PATS eligibility "
            f"criteria ({'; '.join(entries)}).",
        )

    entry = entry_with(entries, _APPOINTMENT_KEYWORDS[kind], rule, "ineligible_situations")
    if kind == "clinical_trial_phase_iv":
        # The PDF excludes trials "other than phase IV trials", so phase IV is the exception.
        if "other than phase iv" not in norm(entry):
            raise RuleParameterError(
                f"{rule.rule_id}: the clinical trial entry no longer says 'other than phase IV', "
                "which this tool relies on"
            )
        return make_hit(
            rule,
            "pass",
            f"Phase IV trials are the exception in '{entry}', so this is not excluded.",
        )
    return make_hit(rule, "fail", f"'{entry}' does not meet PATS eligibility criteria.")


# --- All the gates --------------------------------------------------------------------------


def evaluate_gates(facts: GateFacts, rules: RuleSet) -> GateAssessment:
    """Run every gate and report every hit, so a plan shows all the reasons, not only the first."""
    hits = [
        check_residency(facts, rules),
        *check_apy_lands(facts, rules),
        *check_specialist(facts, rules),
        check_professional(facts, rules),
        check_appointment(facts, rules),
    ]
    return GateAssessment(status=status_for(hits), rule_hits=hits)


# --- decide(): every rule, combined ----------------------------------------------------


class EngineInputError(Exception):
    """The request names a town or service the engine doesn't know, or the two don't match."""


@dataclass(frozen=True)
class EngineContext:
    """The reference data `decide()` needs. Loaded once, then shared (it is never changed)."""

    rules: RuleSet
    towns: dict[str, Town]  # keyed by the town name, lower case
    services: list[ServiceRecord]
    distances: DistanceTable
    # Specialty (lower case) -> the kind of health professional who provides it, for §6.4.4
    #. A specialty that is not listed is a medical specialist.
    professions: dict[str, str]


def load_engine_context(source: ServiceQuery | None = None) -> EngineContext:
    """Load the PATS rules and the synthetic dataset.

    The towns, services and distances come from the dataset files, or from the seeded database if
    a `ServiceQuery` is given. In the app the engine uses the database, the same data the Service
    Discovery Agent reads, so the two can never disagree (a test checks they give the same
    decisions). The §6.4.4 specialty categories always come from the services file.
    """
    if source is None:
        towns, services, distances = load_towns(), load_services(), load_distance_table()
    else:
        towns, services, distances = source.towns(), source.services(), source.distance_table()
    return EngineContext(
        rules=load_rules(),
        towns={norm(town.name): town for town in towns},
        services=services,
        distances=distances,
        professions={
            norm(specialty): category
            for specialty, category in load_non_specialist_categories().items()
        },
    )


def decide(
    goal: GoalRequest,
    facts: ConfirmedFacts,
    destination_service_id: str,
    context: EngineContext,
) -> EligibilityResult:
    """Decide whether the client is eligible for PATS to travel to the chosen service.

    Raises `EngineInputError` for an unknown town or service, and the named errors of the
    individual tools (`DistanceNotFoundError`, `RuleParameterError`) when the data is incomplete.
    The agents catch these, record them in the trace and follow the fallback path.
    """
    rules = context.rules
    destination = next(
        (s for s in context.services if s.service_id == destination_service_id), None
    )
    if destination is None:
        raise EngineInputError(f"destination service {destination_service_id} is not known")
    if norm(goal.specialty) != norm(destination.specialty):
        raise EngineInputError(
            f"destination service {destination.service_id} is for {destination.specialty}, "
            f"not {goal.specialty}"
        )
    origin = context.towns.get(norm(goal.origin_town))
    if origin is None:
        raise EngineInputError(f"origin town {goal.origin_town!r} is not known")

    # A fact confirmed by the client overrides what was extracted from the goal text.
    identity = facts.indigenous_status or goal.indigenous_status
    # The treatment performed matters for §6.5.3; unless one is stated it is the specialty sought.
    treatment = facts.treatment or goal.specialty
    kilometres = context.distances.road_km(origin.town_id, destination.town)

    gates = evaluate_gates(
        GateFacts(
            permanent_sa_resident=facts.permanent_sa_resident,
            residency_situation=facts.residency_situation,
            lives_on_apy_lands=origin.apy_lands,
            indigenous_status=identity,
            specialty=destination.specialty,
            subspecialty=facts.subspecialty,
            professional_category=context.professions.get(norm(destination.specialty)),
            appointment_type=facts.appointment_type,
        ),
        rules,
    )
    away = check_away_from_residence(facts.away_from_residence, rules)
    distance = assess_travel_distance(
        TravelDistanceInput(
            distance_km=kilometres,
            treatment=treatment,
            alternate_route_km=facts.alternate_route_km,
        ),
        rules,
    )
    nearest = assess_nearest_service(
        NearestServiceInput(
            origin_town_id=origin.town_id,
            destination_service_id=destination.service_id,
            bypass_reason=facts.bypass_reason,
            gp_reason_documented=facts.gp_reason_documented,
            local_timeframe_clinically_acceptable=facts.local_timeframe_clinically_acceptable,
            visiting_primary_rooms_service_id=facts.visiting_primary_rooms_service_id,
            referred_from_tertiary_site=facts.referred_from_tertiary_site,
        ),
        context.services,
        context.distances,
        rules,
    )

    hits = [*gates.rule_hits, away, *distance.rule_hits, *nearest.rule_hits]
    outcome = _combine(
        failed=[
            gates.status == "fail",
            away.outcome == "fail",
            distance.status == "not_met",
            nearest.status == "not_met",
        ],
        under_review=[
            gates.status == "needs_review",
            distance.status == "needs_review",
            nearest.status == "needs_review",
        ],
    )

    # Entitlements mean nothing to a client who is not eligible (an excluded APY Lands resident
    # must not be told "subsidies can be paid in advance"), so they are only worked out for one.
    entitlements = None
    if outcome == "eligible":
        entitlements = assess_entitlements(
            EntitlementInput(
                indigenous_status=identity,
                escort_requested=goal.escort_requested,
                client_age=facts.client_age,
                escort_criterion=facts.escort_criterion,
                escort_specialist_approved=facts.escort_specialist_approved,
                concession_card=facts.concession_card,
                severe_financial_hardship=facts.severe_financial_hardship,
                notice_business_days=facts.notice_business_days,
            ),
            rules,
        )
        hits.extend(entitlements.rule_hits)

    return EligibilityResult(
        decision=Decision(
            eligible=outcome == "eligible",
            rule_hits=hits,
            rules_version=rules.guideline_version,
        ),
        outcome=outcome,
        nearest_service_ids=nearest.nearest_service_ids,
        notices=[*distance.notices, *nearest.notices],
        entitlements=entitlements,
    )


def _combine(failed: list[bool], under_review: list[bool]) -> EligibilityOutcome:
    """Any failure makes the client not eligible; otherwise any open review; otherwise eligible."""
    if any(failed):
        return "not_eligible"
    if any(under_review):
        return "needs_review"
    return "eligible"
