"""Nearest-service rules.

Part of the deterministic rules engine. PATS pays travel only to the client's
*nearest* medical specialist service, and resident, visiting and telehealth services all count. This tool answers one question: is the service the client chose the nearest one, and
if not, does a reason in the PDF allow it?

How the PDF's text becomes code:
1. The nearest service is the one with the smallest road distance from the client's home among all
   services for the same specialty, using the distance table. A visiting service always
   counts, whatever the client's dates: if it cannot see the client in time, that
   is a stated clinical fact, and the PDF's own remedy applies (below).
2. A nearer service can be passed in only two ways:
   - the client "is unable to be seen locally within a clinically acceptable timeframe" by a
     visiting specialist, and travels to that specialist's primary rooms (§6.4.6 p. 8); or
   - the client's GP documents one of three criteria (§6.4.6 p. 9). Personal reasons such as
     family support, and "special interest" or skipping a waiting list, never qualify.
3. Whenever a nearer service is passed, the referring practitioner must complete their part of
   the application form (§8.1.1), which the result states.
4. If the test is not met, the plan shows the §6.4.7 "not assessed" notice, so a
   client who might qualify as terminally ill is pointed to a person, not just told "no".

Design choices (the project plan stores the rules as versioned data):
- The lists of reasons and counted service modes come from data/rules/pats_v26_4.yaml. If the data
  loses a reason the code knows about, the tool raises `RuleParameterError`; a missing reason must
  never mean "acceptable".
- "A valid medical reason" is discretionary, so a listed reason without a GP note, and a
  "cannot be treated in SA" referral that does not say where it came from, are `needs_review`
  (an authorised officer, §8.3), not a guess.
"""

from collections.abc import Sequence

from communitybridge.models.dataset import ServiceRecord
from communitybridge.models.eligibility import RuleHit
from communitybridge.models.nearest import (
    NearestServiceAssessment,
    NearestServiceInput,
    NearestStatus,
)
from communitybridge.models.rules import Rule, RuleSet
from communitybridge.tools.distance import DistanceTable
from communitybridge.tools.rule_access import (
    RuleParameterError,
    entry_with,
    find_rule,
    make_hit,
    norm,
    number,
    out_of_scope_notices,
    param,
    require,
    text_list,
    words,
)

NEAREST_RULE = "PATS-6.4.6-NEAREST"
NOT_GROUNDS_RULE = "PATS-6.4.6-NOT-GROUNDS"
NOT_LEGITIMATE_RULE = "PATS-6.4.6-NOT-LEGITIMATE"
PRIMARY_ROOMS_RULE = "PATS-6.4.6-VISITING-PRIMARY-ROOMS"
GP_REASON_RULE = "PATS-6.4.6-GP-REASON"
BYPASS_RULE = "PATS-6.4.6-BYPASS"
REFERRER_RULE = "PATS-8.1.1-REFERRER"

_TERTIARY_REASON = "cannot_be_treated_in_sa"
_TIMEFRAME_REASON = "local_timeframe_clinically_unacceptable"


class NearestServiceError(Exception):
    """The chosen destination is not one of the services given to the tool."""


def candidate_services(
    services: Sequence[ServiceRecord], specialty: str, counted_modes: Sequence[str]
) -> list[ServiceRecord]:
    """The services that count when looking for the nearest one (§6.4.6).

    Same specialty, and a mode the rules data counts: resident, visiting or telehealth. Shared by this rule and the Service Discovery tool, so "nearest" is defined once.
    """
    sought = norm(specialty)
    return [s for s in services if norm(s.specialty) == sought and s.mode in counted_modes]


def rank_by_distance(
    origin_town_id: str, services: Sequence[ServiceRecord], table: DistanceTable
) -> list[tuple[ServiceRecord, float]]:
    """The services with their road distance from the client's home, nearest first.

    Ties are ordered by service id, so the result never depends on the order of the input.
    """
    ranked = [(s, table.road_km(origin_town_id, s.town)) for s in services]
    return sorted(ranked, key=lambda pair: (pair[1], pair[0].service_id))


def _referrer_hit(rules: RuleSet) -> RuleHit:
    """§8.1.1: travelling past the nearest specialist needs the referrer's part of the form."""
    rule = find_rule(rules, REFERRER_RULE)
    entry = entry_with(
        text_list(rule, "referrer_part_required_when"),
        "nearest specialist",
        rule,
        "referrer_part_required_when",
    )
    return make_hit(
        rule,
        "pass",
        f"Because the client {entry}, the referring practitioner must complete the 'Referring "
        "practitioner details' part of the application form (§8.1.1).",
    )


def _classify(reason: str, rules: RuleSet) -> tuple[str, Rule, str]:
    """Is the reason a listed criterion, or one the PDF says is not enough? Refuse if neither.

    Returns ("valid" | "insufficient", the rule it was found in, the entry that matched).
    """
    bypass = find_rule(rules, BYPASS_RULE)
    if reason in text_list(bypass, "valid_bypass_reasons"):
        return "valid", bypass, reason
    for rule_id in (NOT_GROUNDS_RULE, NOT_LEGITIMATE_RULE):
        rule = find_rule(rules, rule_id)
        for entry in text_list(rule, "insufficient_reasons"):
            if words(reason) in words(entry):
                return "insufficient", rule, entry
    raise RuleParameterError(
        f"{BYPASS_RULE}: the reason {reason!r} is in neither the valid reasons nor the "
        "insufficient reasons in the rules data"
    )


def _primary_rooms(
    request: NearestServiceInput, destination: ServiceRecord, nearer: list[ServiceRecord], rules
) -> list[RuleHit] | None:
    """§6.4.6 p. 8: a visiting specialist who cannot see the client in time: primary rooms."""
    rule = find_rule(rules, PRIMARY_ROOMS_RULE)
    if (
        any(service.mode == "visiting" for service in nearer)
        and not request.local_timeframe_clinically_acceptable
        and request.visiting_primary_rooms_service_id == destination.service_id
    ):
        return [
            make_hit(
                rule,
                "pass",
                "The client cannot be seen locally by the visiting specialist within a clinically "
                f"acceptable timeframe, so travel to {destination.name}, the visiting "
                "specialist's primary rooms, is supported (§6.4.6).",
            ),
            _referrer_hit(rules),
        ]
    return None


def _bypass_by_reason(
    request: NearestServiceInput, rules: RuleSet
) -> tuple[NearestStatus, list[RuleHit]]:
    """§6.4.6 p. 8-9: travelling past a nearer service needs a listed reason, documented by a GP."""
    bypass = find_rule(rules, BYPASS_RULE)
    reason = request.bypass_reason
    if reason is None:
        return "not_met", [
            make_hit(
                bypass,
                "fail",
                "No reason for travelling past the nearer service was given. §6.4.6 requires one "
                "of the listed criteria for the claim to be valid.",
            )
        ]

    kind, rule, entry = _classify(reason, rules)
    if kind == "insufficient":
        said = (
            "not sufficient grounds to bypass the nearest service"
            if rule.rule_id == NOT_GROUNDS_RULE
            else "not a legitimate reason to bypass a closer specialist"
        )
        return "not_met", [make_hit(rule, "fail", f"'{entry}' is {said} (§6.4.6).")]

    label = reason.replace("_", " ")
    gp = find_rule(rules, GP_REASON_RULE)
    require(gp, "outcome_if_reason_claimed_but_undocumented", "needs_review")
    if not request.gp_reason_documented:
        return "needs_review", [
            make_hit(
                gp,
                "needs_review",
                f"'{label}' is a listed reason, but the client's GP must provide a valid medical "
                "reason and none is documented, so a PATS authorised officer must review (§8.3).",
            )
        ]

    if reason == _TERTIARY_REASON:
        source = param(bypass, "cannot_be_treated_in_sa_referral_from")
        tertiary = request.referred_from_tertiary_site
        if tertiary is None:
            return "needs_review", [
                make_hit(
                    bypass,
                    "needs_review",
                    f"'{label}' referrals are only accepted from a {source}, and the referrer "
                    "has not been stated, so a PATS authorised officer must review it (§8.3).",
                )
            ]
        if not tertiary:
            return "not_met", [
                make_hit(
                    bypass,
                    "fail",
                    f"'{label}' referrals are only accepted from a {source}, and this one is not "
                    "from one (§6.4.6).",
                )
            ]

    limit = ""
    if reason == _TIMEFRAME_REASON:
        months = number(bypass, "timeframe_bypass_max_months")
        limit = f" PATS subsidies can be paid for up to {months:g} months."
    return "met", [
        make_hit(bypass, "pass", f"'{label}' is one of the listed criteria (§6.4.6).{limit}"),
        make_hit(gp, "pass", "The client's GP has documented a valid medical reason (§6.4.6)."),
        _referrer_hit(rules),
    ]


def assess_nearest_service(
    request: NearestServiceInput,
    services: Sequence[ServiceRecord],
    table: DistanceTable,
    rules: RuleSet,
) -> NearestServiceAssessment:
    """Is the chosen service the client's nearest, and if not, is travelling past it allowed?"""
    nearest_rule = find_rule(rules, NEAREST_RULE)
    by_id = {service.service_id: service for service in services}
    destination = by_id.get(request.destination_service_id)
    if destination is None:
        raise NearestServiceError(
            f"destination service {request.destination_service_id} is not in the services given"
        )

    counted_modes = text_list(nearest_rule, "counted_service_modes")
    candidates = {
        service.service_id: service
        for service in candidate_services(services, destination.specialty, counted_modes)
    }
    # The chosen service is always compared, even if its mode is not one the data counts.
    candidates[destination.service_id] = destination
    ranked = rank_by_distance(request.origin_town_id, list(candidates.values()), table)
    km = {service.service_id: distance for service, distance in ranked}
    nearest_km = ranked[0][1]
    nearest_ids = [service.service_id for service, distance in ranked if distance == nearest_km]
    destination_km = km[destination.service_id]

    def result(status: NearestStatus, hits: list[RuleHit]) -> NearestServiceAssessment:
        notices = []
        if status == "not_met":
            notices = out_of_scope_notices(rules, "nearest_service_requirement_not_met")
        return NearestServiceAssessment(
            status=status,
            nearest_service_ids=nearest_ids,
            nearest_km=nearest_km,
            destination_km=destination_km,
            rule_hits=hits,
            notices=notices,
        )

    modes = ", ".join(counted_modes)
    if destination_km <= nearest_km:  # a tie means the chosen service is one of the nearest
        return result(
            "met",
            [
                make_hit(
                    nearest_rule,
                    "pass",
                    f"{destination.name} ({destination.mode}) is the nearest "
                    f"{destination.specialty} "
                    f"service, {destination_km:g} km away. {modes.capitalize()} services all count "
                    "(§6.4.6).",
                )
            ],
        )

    nearer = [candidates[sid] for sid in nearest_ids]
    names = "; ".join(f"{s.name} ({s.mode}, {km[s.service_id]:g} km)" for s in nearer)
    first = make_hit(
        nearest_rule,
        "fail",
        f"{names} is nearer than {destination.name} ({destination_km:g} km). {modes.capitalize()} "
        "services all count as the nearest service (§6.4.6).",
    )

    rooms = _primary_rooms(request, destination, nearer, rules)
    if rooms is not None:
        return result("met", [first, *rooms])
    status, hits = _bypass_by_reason(request, rules)
    return result(status, [first, *hits])
