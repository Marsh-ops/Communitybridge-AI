"""Distance tool and travel-distance rules.

Serves the rules engine (Eligibility) and, through it, the Eligibility Agent, which
may only explain what this tool decided. It does three jobs:

1. `DistanceTable` looks up the one-way road distance between two towns (§6.5.2). The table is
   the synthetic dataset, a stand-in for the PATS distance calculator.
2. `assess_travel_distance` applies the §6.5 test (more than the threshold, one way) and, for
   trips inside the §6.5.3 "Distance vs Safety" band, the two extra criteria. When the
   requirement isn't met it attaches the §6.5.1 "not assessed" notice.
3. `check_away_from_residence` applies the second §6.5 rule, that travel which starts while
   away from home is not covered.

Design choices, as the project plan asks ("PATS rules stored as versioned data"):
- Every number (100 km, the 90-100 km band, the alternate-route length) and the list of
  lethargic treatments is read from data/rules/pats_v26_4.yaml. Nothing is hard-coded, so a
  guideline update changes the YAML only.
- The code does fix the *meaning* of the tests ("greater than", inclusive band, all criteria
  required). It checks the YAML still says the same and raises `DistanceRuleError` if not, so a
  changed guideline can't be silently applied with the old logic.
- When a fact needed for the band is missing, the result is `needs_review`, not a guess. If one
  criterion has already failed, the answer is fixed, so a missing second fact doesn't matter.
"""

from collections.abc import Iterable
from pathlib import Path

from communitybridge.db.seed import DISTANCES_CSV, load_distances
from communitybridge.models.dataset import Distance
from communitybridge.models.distance import (
    DistanceStatus,
    TravelDistanceAssessment,
    TravelDistanceInput,
)
from communitybridge.models.eligibility import RuleHit, RuleOutcome
from communitybridge.models.rules import Rule, RuleSet
from communitybridge.tools.rule_access import (
    RuleParameterError,
    find_rule,
    make_hit,
    number,
    out_of_scope_notices,
    param,
    require,
)

CALCULATOR_RULE = "PATS-6.5.2-CALCULATOR"
DISTANCE_RULE = "PATS-6.5-DISTANCE"
AWAY_FROM_HOME_RULE = "PATS-6.5-AWAY-FROM-HOME"
SAFETY_RULE = "PATS-6.5.3-SAFETY"

# Inside the §6.5.3 band, the band rule alone decides the requirement (the base §6.5 test has
# already failed by then).
_STATUS_FOR_BAND_OUTCOME: dict[RuleOutcome, DistanceStatus] = {
    "pass": "met",
    "fail": "not_met",
    "needs_review": "needs_review",
}


class DistanceNotFoundError(Exception):
    """The distance table has no figure for the requested pair of towns."""


class DistanceDataError(Exception):
    """The distance table contradicts itself (two different figures for the same pair)."""


# Same class as the shared error, kept under this name for callers that import it from here.
DistanceRuleError = RuleParameterError


# --- §6.5.2 distance table -------------------------------------------------------------------


class DistanceTable:
    """One-way road distances between towns, found in either direction."""

    def __init__(self, distances: Iterable[Distance]) -> None:
        self._km: dict[frozenset[str], float] = {}
        for record in distances:
            key = frozenset((record.from_town_id, record.to_town_id))
            known = self._km.get(key)
            # Keeping either of two different figures would hide a data error, so refuse.
            if known is not None and known != record.road_km:
                raise DistanceDataError(
                    f"conflicting distances for {record.from_town_id} and {record.to_town_id}: "
                    f"{known:g} km and {record.road_km:g} km"
                )
            self._km[key] = record.road_km

    def road_km(self, from_town_id: str, to_town_id: str) -> float:
        """Road distance in km between two towns, the same either way round (§6.5.2).

        Raises `DistanceNotFoundError` for a pair missing from the table, so a gap in the data is
        never read as a short trip.
        """
        if from_town_id == to_town_id:
            return 0.0
        try:
            return self._km[frozenset((from_town_id, to_town_id))]
        except KeyError:
            raise DistanceNotFoundError(
                f"no distance for {from_town_id} to {to_town_id} in the distance table"
            ) from None


def load_distance_table(path: Path = DISTANCES_CSV) -> DistanceTable:
    """Build the table from the synthetic distance file (the same one seed_db.py loads)."""
    return DistanceTable(load_distances(path))


# --- §6.5 and §6.5.3 -------------------------------------------------------------------------


def _assess_band(
    request: TravelDistanceInput, safety: Rule, low: float, high: float
) -> tuple[RuleOutcome, str]:
    """§6.5.3: both criteria must be met. Returns the outcome and a plain-language reason."""
    alternate_over = number(safety, "alternate_route_km_over")
    lethargic = [str(item) for item in param(safety, "lethargic_treatments")]
    treatment = (request.treatment or "").strip()
    alternate = request.alternate_route_km

    # None = not known yet. A match is exact (ignoring case and spaces): "i.e. chemotherapy or
    # radiation" is a closed list, so similar words are not guessed at.
    alternate_ok = None if alternate is None else alternate > alternate_over
    treatment_ok = (
        None if not treatment else treatment.casefold() in {t.casefold() for t in lethargic}
    )

    band = f"between {low:g} and {high:g} km"
    failed = []
    if alternate_ok is False:
        failed.append(f"the alternate route ({alternate:g} km) is not over {alternate_over:g} km")
    if treatment_ok is False:
        listed = " or ".join(lethargic)
        failed.append(f"{treatment} is not a treatment that leaves the client lethargic ({listed})")
    if failed:
        reason = f"For a trip {band} both criteria must be met, but {' and '.join(failed)}."
        return "fail", reason

    missing = []
    if alternate_ok is None:
        missing.append("the alternate route distance")
    if treatment_ok is None:
        missing.append("the treatment")
    if missing:
        verb = "is" if len(missing) == 1 else "are"
        return (
            "needs_review",
            f"For a trip {band} both criteria must be checked, but {' and '.join(missing)} "
            f"{verb} not known, so a PATS authorised officer must review it.",
        )

    return (
        "pass",
        f"The trip is {band}, the alternate route ({alternate:g} km) is over "
        f"{alternate_over:g} km and {treatment} is a treatment that leaves the client lethargic.",
    )


def assess_travel_distance(
    request: TravelDistanceInput, rules: RuleSet
) -> TravelDistanceAssessment:
    """Apply §6.5 (and §6.5.3 inside its band) to one trip, citing §6.5.2 for the distance."""
    calculator = find_rule(rules, CALCULATOR_RULE)
    gate = find_rule(rules, DISTANCE_RULE)
    safety = find_rule(rules, SAFETY_RULE)

    require(gate, "comparison", "greater_than")
    require(safety, "bounds_inclusive", True)
    require(safety, "all_criteria_required", True)
    threshold = number(gate, "threshold_km")
    low = number(safety, "min_km")
    high = number(safety, "max_km")
    km = request.distance_km

    hits = [
        make_hit(
            calculator,
            "pass",
            f"Distance of {km:g} km taken from the synthetic distance table, the prototype's "
            "stand-in for the PATS distance calculator.",
        )
    ]
    basis = param(gate, "distance_basis")

    if km > threshold:
        hits.append(make_hit(gate, "pass", f"{km:g} km is more than {threshold:g} km ({basis})."))
        return TravelDistanceAssessment(distance_km=km, status="met", rule_hits=hits)

    in_band = low <= km <= high
    next_step = f"; the {low:g} to {high:g} km band is checked next" if in_band else ""
    hits.append(
        make_hit(gate, "fail", f"{km:g} km is not more than {threshold:g} km ({basis}){next_step}.")
    )

    status: DistanceStatus = "not_met"
    if in_band:
        outcome, reason = _assess_band(request, safety, low, high)
        hits.append(make_hit(safety, outcome, reason))
        status = _STATUS_FOR_BAND_OUTCOME[outcome]

    notices = (
        out_of_scope_notices(rules, "distance_requirement_not_met") if status == "not_met" else []
    )
    return TravelDistanceAssessment(distance_km=km, status=status, rule_hits=hits, notices=notices)


def check_away_from_residence(away_from_residence: bool, rules: RuleSet) -> RuleHit:
    """§6.5: travel that starts while away from home (e.g. on holidays) is not covered."""
    rule = find_rule(rules, AWAY_FROM_HOME_RULE)
    if away_from_residence:
        return make_hit(
            rule,
            "fail",
            "The client is travelling to treatment while away from their residence (for example "
            "on holidays or interstate), which PATS does not cover.",
        )
    return make_hit(rule, "pass", "The trip starts from the client's permanent residence.")
