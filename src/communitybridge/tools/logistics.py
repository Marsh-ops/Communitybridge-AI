"""Logistics calculator: travel mode, accommodation and excluded modes.

Serves the Logistics Agent. It runs after the rules engine has said the client is
eligible, and it never decides eligibility. Each function answers one question the
plan needs, and cites the PATS section behind every figure:

- `compare_modes`: which way of travelling PATS pays for. PATS pays for "the most economical mode
  of travel appropriate for the client's functional ability and medical condition" (§7.2). The
  surface cost is the cheaper of a private vehicle at the printed per-km rate (§7.2.1) and a
  standard public transport fare per traveller (§7.2.2). Flying is subsidised in three situations
  (§7.2.5): (1) the treating specialist authorises it under a listed medical criterion (§7.2.6);
  (2) surface travel plus two nights of accommodation subsidy costs more than return air travel;
  (3) the local clinician authorises it because public transport doesn't suit.
- `accommodation_subsidy`: what PATS pays towards a stay (§7.3, §7.3.1), and only in commercial
  accommodation, never with family or friends (§6.6.1).
- `flag_ineligible_modes`: which modes the client asked about that PATS does not subsidise
  (§6.5.8).
- `assess_logistics`: all of the above for one eligible client.

Choices made when the PDF leaves a gap (none of these invent a PATS rule):
- Of the two nights in the §7.2.5 comparison, only those §7.3.1 would actually subsidise are
  counted, at the $110 or $154 caps.
- Item 3 of §7.2.5 ("may be authorised") is discretionary, so it is `needs_review`. It
  only comes up when there is no surface option at all.
- A stay the treating specialist has authorised beyond the automatic two nights is `needs_review`:
  the PDF doesn't say how the first-night rule applies to authorised nights.
- Where an answer depends on whether the client identifies as Aboriginal and/or Torres Strait
  Islander and they haven't said, the plan asks and shows the answer as "depends".

All fares come from the SIMULATED fare sheet; every rate and limit comes from the rules data.
"""

from dataclasses import dataclass

from communitybridge.models.eligibility import RuleHit, RuleOutcome
from communitybridge.models.engine import EligibilityResult
from communitybridge.models.entitlements import EntitlementStatus
from communitybridge.models.facts import ConfirmedFacts
from communitybridge.models.logistics import (
    AccommodationSubsidy,
    AccommodationSubsidyInput,
    AirBasis,
    LogisticsAssessment,
    ModeComparison,
    ModeComparisonInput,
    ModeCosts,
    ModeFlags,
    SurfaceMode,
    TravelMode,
)
from communitybridge.models.request import GoalRequest
from communitybridge.models.rules import Rule, RuleSet
from communitybridge.models.simulated import FareSheet, RouteFare
from communitybridge.tools.entitlements import IDENTITY_QUESTION
from communitybridge.tools.rule_access import (
    RuleParameterError,
    entry_with,
    find_rule,
    make_hit,
    norm,
    number,
    param,
    require,
    text_list,
)
from communitybridge.tools.rules_engine import EngineContext

MOST_ECONOMICAL_RULE = "PATS-7.2-MOST-ECONOMICAL"
VEHICLE_RULE = "PATS-7.2.1-PRIVATE-VEHICLE"
PUBLIC_RULE = "PATS-7.2.2-PUBLIC-TRANSPORT"
AIR_ECONOMICAL_RULE = "PATS-7.2.5-AIR-ECONOMICAL"
AIR_MEDICAL_RULE = "PATS-7.2.5-AIR-MEDICAL"
AIR_NO_PT_RULE = "PATS-7.2.5-AIR-NO-SUITABLE-PT"
SURCHARGES_RULE = "PATS-7.2.5-SURCHARGES"
AIR_CRITERIA_RULE = "PATS-7.2.6-AIR-CRITERIA"
RATES_RULE = "PATS-7.3-RATES"
NIGHTS_RULE = "PATS-7.3.1-NIGHTS"
COMMERCIAL_RULE = "PATS-6.6.1-COMMERCIAL"
FAMILY_RULE = "PATS-6.6.1-FAMILY"
NON_ELIGIBLE_RULE = "PATS-6.5.8-NON-ELIGIBLE-TRAVEL"


class LogisticsError(Exception):
    """The request can't be planned: the client isn't eligible, or a town or service is unknown."""


# --- Small helpers ---------------------------------------------------------------------------


def _money(amount: float) -> float:
    return round(amount, 2)


def _usd(amount: float) -> str:
    return f"${amount:,.2f}"


@dataclass(frozen=True)
class _Rates:
    rule: Rule
    alone: float  # $110 per person per night
    with_escort: float  # $154 per night for a client and an authorised escort

    def cap(self, travellers: int) -> float:
        return self.with_escort if travellers >= 2 else self.alone


def _read_rates(rules: RuleSet) -> _Rates:
    rule = find_rule(rules, RATES_RULE)
    require(rule, "pay_actual_cost_if_lower", True)  # the code pays min(actual cost, cap)
    return _Rates(
        rule,
        number(rule, "max_per_person_per_night_aud"),
        number(rule, "max_patient_and_escort_per_night_aud"),
    )


def _nights_subsidised(total_nights: int, first_night_covered: bool, nights_rule: Rule) -> int:
    """How many of `total_nights` §7.3.1 would subsidise.

    A concession card holder is subsidised from the first night. Without one, the client pays the
    first night in full, unless the Aboriginal Engagement Strategy waives the concession card
    (§6.1.2), in which case it counts as if they held one.
    """
    if first_night_covered:
        first_night_paid = bool(param(nights_rule, "concession_first_night_subsidised"))
    else:
        first_night_paid = bool(param(nights_rule, "non_concession_first_night_subsidised"))
    return total_nights if first_night_paid else max(total_nights - 1, 0)


def _trip_cost(route: RouteFare, travellers: int) -> float:
    """A bus fare is one way per person (so two legs); an air fare is already a return."""
    legs = 2 if route.fare_type == "one_way" else 1
    return _money(route.fare_aud * travellers * legs)


def _cheapest(routes: list[RouteFare], travellers: int) -> tuple[RouteFare | None, float | None]:
    if not routes:
        return None, None
    best = min(routes, key=lambda route: _trip_cost(route, travellers))
    return best, _trip_cost(best, travellers)


# --- §7.3.1 the first night, on its own ----------------------------------------------------------


def first_night_subsidised(
    concession_card: bool, first_night_waiver: EntitlementStatus, rules: RuleSet
) -> tuple[bool | None, RuleHit]:
    """Is the first night subsidised? None means it depends on an answer the client hasn't given."""
    rule = find_rule(rules, NIGHTS_RULE)
    if concession_card:
        return True, make_hit(
            rule, "pass", "A concession card holder is subsidised from the first night (§7.3.1)."
        )
    if first_night_waiver == "entitled":
        return True, make_hit(
            rule,
            "pass",
            "The first night is subsidised without a concession card under the Aboriginal "
            "Engagement Strategy (§6.1.2, §7.3.1).",
        )
    if first_night_waiver == "ask":
        return None, make_hit(
            rule,
            "not_applicable",
            "Whether the first night is subsidised depends on whether the client identifies as "
            "Aboriginal and/or Torres Strait Islander, and they have not said (§6.1.2, §7.3.1).",
        )
    return False, make_hit(
        rule,
        "not_applicable",
        "Without a concession card the client pays the first night in full; the second night is "
        "subsidised (§7.3.1).",
    )


# --- §7.2, §7.2.1, §7.2.2, §7.2.5, §7.2.6: the most economical mode ---------------------------


@dataclass(frozen=True)
class _ModeRules:
    """The rules `compare_modes` reads, checked once at the start."""

    most: Rule
    vehicle: Rule
    public: Rule
    economical: Rule
    nights: Rule
    rates: _Rates
    public_modes: set[str]
    comparison_nights: int


def public_transport_modes(rules: RuleSet) -> set[str]:
    """The public surface modes §7.2 lists (bus, rail, passenger ferry), without the private car."""
    most = find_rule(rules, MOST_ECONOMICAL_RULE)
    surface_modes = text_list(most, "surface_modes")
    entry_with(surface_modes, "bus", most, "surface_modes")  # public transport must be listed
    return {norm(mode) for mode in surface_modes} - {"car"}


def _load_mode_rules(rules: RuleSet) -> _ModeRules:
    economical = find_rule(rules, AIR_ECONOMICAL_RULE)
    # The code implements these readings of §7.2.5 item 2; refuse data that says otherwise.
    require(economical, "nights_counted", "subsidised_only")
    require(economical, "medical_endorsement_required", False)
    require(economical, "include_escort_if_authorised", True)
    return _ModeRules(
        most=find_rule(rules, MOST_ECONOMICAL_RULE),
        vehicle=find_rule(rules, VEHICLE_RULE),
        public=find_rule(rules, PUBLIC_RULE),
        economical=economical,
        nights=find_rule(rules, NIGHTS_RULE),
        rates=_read_rates(rules),
        public_modes=public_transport_modes(rules),
        comparison_nights=int(number(economical, "comparison_accommodation_nights")),
    )


@dataclass(frozen=True)
class _Prices:
    """What each way of travelling would cost for the whole return journey."""

    vehicle: float | None
    public: float | None
    bus: RouteFare | None
    air: float | None
    surface: float | None
    surface_mode: SurfaceMode | None


def _price_the_options(request: ModeComparisonInput, mr: _ModeRules) -> _Prices:
    vehicle = None
    if request.has_private_vehicle:
        per_km = number(mr.vehicle, "rate_cents_per_km") / 100
        vehicle = _money(2 * request.road_km * per_km)  # return trip; one vehicle, one application
    buses = [r for r in request.fares.routes if r.mode in mr.public_modes]
    bus, public = _cheapest(buses, request.travellers)
    _, air = _cheapest([r for r in request.fares.routes if r.mode == "air"], request.travellers)
    surface = min((cost for cost in (vehicle, public) if cost is not None), default=None)
    surface_mode: SurfaceMode | None = None
    if surface is not None:
        surface_mode = "private_vehicle" if surface == vehicle else "public_transport"
    return _Prices(vehicle, public, bus, air, surface, surface_mode)


@dataclass(frozen=True)
class _Reading:
    """The surface-versus-air sums for one reading of the first-night rule."""

    nights_counted: int
    accommodation: float
    surface_total: float | None
    air_economical: bool


def _read_the_sums(
    prices: _Prices, mr: _ModeRules, travellers: int, first_night_covered: bool
) -> _Reading:
    nights = _nights_subsidised(mr.comparison_nights, first_night_covered, mr.nights)
    accommodation = _money(nights * mr.rates.cap(travellers))
    total = _money(prices.surface + accommodation) if prices.surface is not None else None
    # §7.2.5 item 2: air is the most economical only if surface travel plus the nights costs
    # MORE than return air travel. Equal is not more.
    flies = prices.air is not None and total is not None and total > prices.air
    return _Reading(nights, accommodation, total, flies)


def _medical_air(
    request: ModeComparisonInput, rules: RuleSet, air_available: bool
) -> tuple[list[RuleHit], bool]:
    """§7.2.5 item 1 and §7.2.6: the treating specialist authorises air under a listed criterion."""
    if request.air_criterion is None:
        return [], False
    criteria_rule = find_rule(rules, AIR_CRITERIA_RULE)
    criterion = next(
        (c for c in text_list(criteria_rule, "criteria") if norm(c) == norm(request.air_criterion)),
        None,
    )
    if criterion is None:  # a missing criterion must not quietly mean "not required"
        raise RuleParameterError(
            f"{criteria_rule.rule_id}: criterion {request.air_criterion!r} is not in the data"
        )
    authorised = request.air_specialist_authorised
    if authorised:
        authorisation = f"The treating medical specialist has authorised air under '{criterion}'."
    else:
        authorisation = (
            f"Air travel under '{criterion}' must be authorised by the treating medical "
            "specialist on the application form, and it has not been."
        )
    hits = [
        make_hit(
            criteria_rule, "pass" if authorised else "not_applicable", f"{authorisation} (§7.2.6)"
        )
    ]
    medical = find_rule(rules, AIR_MEDICAL_RULE)
    granted = authorised and air_available
    if granted:
        hits.append(
            make_hit(
                medical,
                "pass",
                "The client can't travel by surface mode for medical reasons, so air travel is "
                "subsidised whatever it costs (§7.2.5 item 1).",
            )
        )
    else:
        why = (
            "There is no air route between these towns (SIMULATED timetable)."
            if authorised
            else "The specialist's authorisation is missing."
        )
        hits.append(make_hit(medical, "not_applicable", f"Medical air travel not granted. {why}"))
    return hits, granted


def _headline_hit(mr: _ModeRules, chosen: TravelMode | None, depends: bool) -> RuleHit:
    """The §7.2 answer in a sentence: the mode, or why there isn't one yet."""
    if chosen is not None:
        how = {"air": "air travel", "private_vehicle": "a private vehicle"}.get(
            chosen, "public transport"
        )
        simulated = "" if chosen == "private_vehicle" else " (SIMULATED fares)"
        return make_hit(
            mr.most,
            "pass",
            f"The most economical mode appropriate for the client is {how}{simulated} (§7.2).",
        )
    if depends:
        return make_hit(
            mr.most,
            "not_applicable",
            "The most economical mode depends on whether the client identifies as Aboriginal "
            "and/or Torres Strait Islander, and they have not said, so the plan asks (§7.2).",
        )
    return make_hit(
        mr.most,
        "needs_review",
        "No private vehicle or public transport route is available, and no flight can be chosen "
        "without authorisation, so there is no mode to calculate (§7.2).",
    )


def _vehicle_hit(request: ModeComparisonInput, mr: _ModeRules, prices: _Prices) -> RuleHit:
    if prices.vehicle is None:
        return make_hit(mr.vehicle, "not_applicable", "The client has no private vehicle (§7.2.1).")
    cents = number(mr.vehicle, "rate_cents_per_km")
    return make_hit(
        mr.vehicle,
        "pass",
        f"Private vehicle: 2 x {request.road_km:g} km x {cents:g} cents per km = "
        f"{_usd(prices.vehicle)}. Only one application is eligible per vehicle (§7.2.1).",
    )


def _public_hit(request: ModeComparisonInput, mr: _ModeRules, prices: _Prices) -> RuleHit:
    if prices.bus is None or prices.public is None:
        return make_hit(
            mr.public,
            "not_applicable",
            "There is no public transport route between these towns (SIMULATED) (§7.2.2).",
        )
    fare = f"{_usd(prices.bus.fare_aud)} {prices.bus.fare_type.replace('_', ' ')}"
    return make_hit(
        mr.public,
        "pass",
        f"Public transport on {prices.bus.route_id} (SIMULATED fare {fare}) for "
        f"{request.travellers} traveller(s) costs {_usd(prices.public)} return, on a standard "
        "priced ticket (§7.2.2).",
    )


def _air_hit(mr: _ModeRules, prices: _Prices, shown: _Reading, depends: bool) -> RuleHit:
    """§7.2.5 item 2: surface travel plus the nights, against return air travel."""
    if prices.air is None:
        return make_hit(
            mr.economical,
            "not_applicable",
            "There is no air route between these towns (SIMULATED timetable) (§7.2.5).",
        )
    if depends:
        return make_hit(
            mr.economical,
            "not_applicable",
            "Whether flying is the most economical mode depends on whether the client identifies "
            "as Aboriginal and/or Torres Strait Islander, and they have not said (§7.2.5 item 2).",
        )
    if shown.surface_total is None or prices.surface is None:
        return make_hit(
            mr.economical,
            "not_applicable",
            "There is no surface option to compare with return air travel (§7.2.5).",
        )
    sums = (
        f"Surface travel ({_usd(prices.surface)}) plus {shown.nights_counted} subsidised "
        f"night(s) ({_usd(shown.accommodation)}) costs {_usd(shown.surface_total)}"
    )
    if shown.air_economical:
        return make_hit(
            mr.economical,
            "pass",
            f"{sums}, more than return air travel ({_usd(prices.air)}), so air travel is the "
            "most economical mode. A medical endorsement for air travel is not required in this "
            "case (§7.2.5 item 2).",
        )
    return make_hit(
        mr.economical,
        "not_applicable",
        f"{sums}, which is not more than return air travel ({_usd(prices.air)}), so air is not "
        "the cheapest mode (§7.2.5 item 2).",
    )


def _rate_and_night_hits(mr: _ModeRules, shown: _Reading, travellers: int) -> list[RuleHit]:
    rates = mr.rates
    full = shown.nights_counted == mr.comparison_nights
    paid_by_client = (
        "" if full else " (the client pays the first night in full without a concession card)"
    )
    return [
        make_hit(
            rates.rule,
            "pass",
            f"Accommodation is subsidised up to {_usd(rates.alone)} per person per night, or "
            f"{_usd(rates.with_escort)} for a client and an authorised escort; "
            f"{_usd(rates.cap(travellers))} a night is used here (§7.3).",
        ),
        make_hit(
            mr.nights,
            "pass" if full else "not_applicable",
            f"Of {mr.comparison_nights} nights, {shown.nights_counted} would be subsidised"
            f"{paid_by_client} (§7.3.1).",
        ),
    ]


def compare_modes(request: ModeComparisonInput, rules: RuleSet) -> ModeComparison:
    """Which way of travelling PATS pays for, and the figures behind it (SIMULATED fares)."""
    mr = _load_mode_rules(rules)
    prices = _price_the_options(request, mr)

    # The first-night rule is known, or it is an unanswered question about the client's identity.
    covered = request.concession_card or request.first_night_waiver == "entitled"
    unanswered = not covered and request.first_night_waiver == "ask"
    readings = [
        _read_the_sums(prices, mr, request.travellers, first_night_covered)
        for first_night_covered in ([False, True] if unanswered else [covered])
    ]
    depends = len({reading.air_economical for reading in readings}) > 1
    shown = readings[0]  # when it depends, show the cautious figures (the client pays night one)

    medical_hits, medical_air = _medical_air(request, rules, prices.air is not None)
    chosen: TravelMode | None
    basis: AirBasis | None = None
    if medical_air:
        chosen, basis = "air", "medical"
    elif prices.surface is None or depends:
        chosen = None
    elif shown.air_economical:
        chosen, basis = "air", "economical"
    else:
        chosen = prices.surface_mode

    # What PATS would pay for the chosen mode's travel (the nights are worked out separately).
    cost_of_chosen = {
        "air": prices.air,
        "private_vehicle": prices.vehicle,
        "public_transport": prices.public,
    }.get(chosen or "")

    hits = [
        _headline_hit(mr, chosen, depends),
        *medical_hits,
        _vehicle_hit(request, mr, prices),
        _public_hit(request, mr, prices),
        _air_hit(mr, prices, shown, depends),
        *_rate_and_night_hits(mr, shown, request.travellers),
    ]
    if chosen is None and prices.surface is None and prices.air is not None and not medical_air:
        # §7.2.5 item 3: no suitable public transport, so flying "may be authorised".
        hits.append(
            make_hit(
                find_rule(rules, AIR_NO_PT_RULE),
                "needs_review",
                "There is no surface option, so air travel may be authorised, but that "
                "authorisation must come from a regional LHN local authorising clinician "
                "(§7.2.5 item 3).",
            )
        )
    surcharges = find_rule(rules, SURCHARGES_RULE)
    excluded = ", ".join(text_list(surcharges, "excluded_costs"))
    hits.append(
        make_hit(
            surcharges,
            "pass",
            f"Only the actual fare is subsidised. Surcharges such as {excluded} are not (§7.2.5).",
        )
    )

    return ModeComparison(
        chosen_mode=chosen,
        air_basis=basis,
        air_most_economical=None if depends else (basis == "economical" or shown.air_economical),
        surface_mode=prices.surface_mode,
        travel_subsidy_aud=cost_of_chosen,
        costs=ModeCosts(
            private_vehicle_aud=prices.vehicle,
            public_transport_aud=prices.public,
            surface_aud=prices.surface,
            nights_counted=shown.nights_counted,
            accommodation_aud=shown.accommodation,
            surface_total_aud=shown.surface_total,
            air_aud=prices.air,
        ),
        questions=[IDENTITY_QUESTION] if depends else [],
        rule_hits=hits,
    )


# --- §6.6.1, §7.3, §7.3.1: accommodation -------------------------------------------------------


def accommodation_subsidy(
    request: AccommodationSubsidyInput, rules: RuleSet
) -> AccommodationSubsidy:
    """What PATS pays towards one stay: up to the nightly cap, for the nights §7.3.1 covers."""
    rates = _read_rates(rules)
    nights_rule = find_rule(rules, NIGHTS_RULE)
    commercial_rule = find_rule(rules, COMMERCIAL_RULE)
    family_rule = find_rule(rules, FAMILY_RULE)
    cap = rates.cap(request.travellers)
    automatic = int(number(nights_rule, "nights_without_medical_authorisation"))
    first_night, first_night_hit = first_night_subsidised(
        request.concession_card, request.first_night_waiver, rules
    )

    who = "a client and an authorised escort" if request.travellers >= 2 else "one person"
    hits = [
        make_hit(
            rates.rule,
            "pass",
            f"Subsidy is up to {_usd(cap)} a night for {who}; if the accommodation costs less, "
            "the actual cost is paid (§7.3).",
        ),
        make_hit(
            commercial_rule,
            "pass" if request.commercial else "fail",
            "The accommodation is commercial (it has an ABN) (§6.6.1)."
            if request.commercial
            else "The accommodation is not commercial (no ABN, or a rehabilitation, primary care "
            "or supported residential facility), so there is no accommodation subsidy (§6.6.1).",
        ),
        make_hit(
            family_rule,
            "fail" if request.staying_with_family_or_friends else "pass",
            "There is no subsidy for a client who stays with family or friends (§6.6.1)."
            if request.staying_with_family_or_friends
            else "The client is not staying with family or friends (§6.6.1).",
        ),
    ]

    def result(**fields) -> AccommodationSubsidy:
        return AccommodationSubsidy(
            first_night_subsidised=first_night, per_night_cap_aud=cap, rule_hits=hits, **fields
        )

    if not request.commercial or request.staying_with_family_or_friends:
        hits.append(first_night_hit)
        return result(status="calculated", nights_subsidised=0, maximum_aud=0.0, subsidy_aud=0.0)

    if request.specialist_authorised_nights is not None and request.nights > automatic:
        hits.append(
            make_hit(
                nights_rule,
                "needs_review",
                "The treating specialist has authorised a longer stay. The automatic two nights "
                "are not in addition to authorised nights, and the guideline does not say how "
                "the first-night rule applies to them, so a PATS authorised officer must review "
                "it (§7.3.1, §8.3).",
            )
        )
        return result(status="needs_review")

    eligible_nights = min(request.nights, automatic)
    covered = first_night is True
    if first_night is None:  # depends on an unanswered identity question
        with_waiver = _nights_subsidised(eligible_nights, True, nights_rule)
        without = _nights_subsidised(eligible_nights, False, nights_rule)
        if with_waiver != without:
            hits.append(first_night_hit)
            return result(status="ask", questions=[IDENTITY_QUESTION])
    nights = _nights_subsidised(eligible_nights, covered, nights_rule)

    extra = ""
    if request.nights > automatic:
        extra = (
            f" Only the first {automatic} nights are automatic; later nights need the treating "
            "specialist to authorise the length of stay."
        )
    hits.append(
        make_hit(
            nights_rule,
            "pass" if nights else "not_applicable",
            f"{nights} of {request.nights} night(s) are subsidised.{extra} "
            f"{first_night_hit.reason}",
        )
    )
    per_night = min(request.nightly_cost_aud, cap) if request.nightly_cost_aud is not None else None
    return result(
        status="calculated",
        nights_subsidised=nights,
        maximum_aud=_money(nights * cap),
        subsidy_aud=_money(nights * per_night) if per_night is not None else None,
    )


# --- §6.5.8: modes PATS does not subsidise -----------------------------------------------------


def _alternatives(entry: str) -> set[str]:
    """'Uber or other ride sharing platform' can be asked about as 'Uber', so split it up."""
    return {norm(entry)} | {norm(part) for part in entry.split(" or ")}


def flag_ineligible_modes(requested_modes: list[str], rules: RuleSet) -> ModeFlags:
    """Which of the modes the client asked about PATS does not subsidise (§6.5.8).

    Names are matched whole: an ordinary "bus" is not a "shuttle bus". A mode the tool recognises
    as neither excluded nor an ordinary eligible one is reported as `unrecognised`, never as fine.
    """
    if not requested_modes:
        return ModeFlags()
    rule = find_rule(rules, NON_ELIGIBLE_RULE)
    excluded = {
        alt for entry in text_list(rule, "ineligible_modes") for alt in _alternatives(entry)
    }
    most = find_rule(rules, MOST_ECONOMICAL_RULE)
    air_title = find_rule(rules, AIR_ECONOMICAL_RULE).title
    eligible = {norm(mode) for mode in text_list(most, "surface_modes")} | {"air", norm(air_title)}

    flagged: list[str] = []
    unrecognised: list[str] = []
    seen: set[str] = set()
    for mode in requested_modes:
        key = norm(mode)
        if not key or key in seen:
            continue
        seen.add(key)
        if key in excluded:
            flagged.append(mode.strip())
        elif key not in eligible:
            unrecognised.append(mode.strip())

    outcome: RuleOutcome
    if flagged:
        outcome = "fail"
        reason = (
            f"PATS does not subsidise: {', '.join(flagged)} (§6.5.8). Only the primary modes of "
            "travel between the client's home and the treatment location are subsidised."
        )
    elif unrecognised:
        outcome = "not_applicable"
        reason = (
            "These modes were not recognised, so they have not been assessed: "
            f"{', '.join(unrecognised)} (§6.5.8)."
        )
    else:
        outcome = "pass"
        reason = "None of the modes asked about is excluded by §6.5.8."
    return ModeFlags(
        flagged=flagged,
        unrecognised=unrecognised,
        rule_hits=[make_hit(rule, outcome, reason)],
    )


# --- Everything for one eligible client ---------------------------------------------------------


def assess_logistics(
    result: EligibilityResult,
    goal: GoalRequest,
    facts: ConfirmedFacts,
    destination_service_id: str,
    fares: FareSheet,
    context: EngineContext,
) -> LogisticsAssessment:
    """Plan the travel for a client the rules engine found eligible.

    Raises `LogisticsError` if the client isn't eligible (there is nothing to plan, and a plan must
    never imply a subsidy for someone who isn't entitled to one) or a town or service is unknown.
    """
    if result.outcome != "eligible" or result.entitlements is None:
        raise LogisticsError("logistics are only planned for a client the engine found eligible")
    destination = next(
        (s for s in context.services if s.service_id == destination_service_id), None
    )
    origin = context.towns.get(norm(goal.origin_town))
    if destination is None or origin is None:
        raise LogisticsError("the destination service or the origin town is not known")

    entitlements = result.entitlements
    travellers = 1 + (1 if entitlements.escort == "entitled" else 0)
    modes = compare_modes(
        ModeComparisonInput(
            road_km=context.distances.road_km(origin.town_id, destination.town),
            travellers=travellers,
            has_private_vehicle=facts.has_private_vehicle,
            concession_card=facts.concession_card,
            first_night_waiver=entitlements.first_night_concession_waived,
            air_criterion=facts.air_medical_criterion,
            air_specialist_authorised=facts.air_specialist_authorised,
            fares=fares,
        ),
        context.rules,
    )
    first_night, _ = first_night_subsidised(
        facts.concession_card, entitlements.first_night_concession_waived, context.rules
    )
    flags = flag_ineligible_modes(facts.requested_modes, context.rules)
    questions = list(dict.fromkeys([*entitlements.questions, *modes.questions]))
    return LogisticsAssessment(
        travellers=travellers,
        modes=modes,
        first_night_subsidised=first_night,
        ineligible_modes_flagged=flags.flagged,
        unrecognised_modes=flags.unrecognised,
        questions=questions,
        rule_hits=[*modes.rule_hits, *flags.rule_hits],
    )
