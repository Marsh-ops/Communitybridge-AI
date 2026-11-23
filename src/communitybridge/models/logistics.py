"""Models for the logistics calculator: travel mode, accommodation and non-eligible modes.

The logistics calculator works out what a journey costs and which way of travelling PATS pays for.
It runs after the rules engine has said the client is eligible, so it never decides eligibility. Like the other tools it takes validated inputs and returns read-only results.

Money is in Australian dollars, rounded to the cent. Every cost uses a fare or rate from a rule or
a SIMULATED fare sheet, never an invented number.

`None` in a result means "depends on an unanswered question", not "no". It follows this rule:
where the answer turns on whether the client identifies as Aboriginal and/or Torres
Strait Islander and they have not said, the plan asks and shows the answer as "depends".
"""

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from communitybridge.models.common import NonEmptyStr, StrictModel
from communitybridge.models.eligibility import RuleHit
from communitybridge.models.entitlements import EntitlementStatus
from communitybridge.models.simulated import FareSheet

# §7.2.6: the four criteria the treating specialist can authorise medical air travel under.
AirCriterion = Literal[
    "Active clinical management",
    "Pain management",
    "Clinical urgency",
    "Restricted mobility",
]
SurfaceMode = Literal["private_vehicle", "public_transport"]
TravelMode = Literal["private_vehicle", "public_transport", "air"]
# Why air is the mode: it is the most economical (§7.2.5 item 2), or the treating specialist
# authorised it for medical reasons (§7.2.5 item 1, §7.2.6).
AirBasis = Literal["economical", "medical"]


# --- Travel mode (§7.2, §7.2.1, §7.2.2, §7.2.5, §7.2.6) ------------------------------------


class ModeComparisonInput(StrictModel):
    """What the most-economical-mode calculation needs for one return journey."""

    model_config = ConfigDict(frozen=True)

    road_km: float = Field(ge=0)  # one way, from the distance table (§6.5.2)
    travellers: int = Field(ge=1)  # the client, plus an escort if one is entitled (§7.2.8)
    has_private_vehicle: bool = True  # a client without a car cannot claim the §7.2.1 rate
    concession_card: bool = False
    # §6.1.2: Aboriginal and Torres Strait Islander clients don't need a concession card for
    # the first night. "ask" means the client hasn't said, so the answer may depend on it.
    first_night_waiver: EntitlementStatus = "not_entitled"
    air_criterion: AirCriterion | None = None
    air_specialist_authorised: bool = False
    fares: FareSheet


class ModeCosts(StrictModel):
    """The figures behind the comparison. None means that option is not available."""

    model_config = ConfigDict(frozen=True)

    private_vehicle_aud: float | None = Field(default=None, ge=0)
    public_transport_aud: float | None = Field(default=None, ge=0)
    surface_aud: float | None = Field(default=None, ge=0)  # the cheaper of the two above
    nights_counted: int = Field(ge=0)  # of the two nights, those §7.3.1 would subsidise
    accommodation_aud: float = Field(ge=0)
    surface_total_aud: float | None = Field(default=None, ge=0)  # surface travel + the nights
    air_aud: float | None = Field(default=None, ge=0)


class ModeComparison(StrictModel):
    """Which way of travelling PATS pays for, and why. Read-only."""

    model_config = ConfigDict(frozen=True)

    chosen_mode: TravelMode | None  # None: depends on an unanswered question, or needs review
    air_basis: AirBasis | None
    # True/False once known; None when it depends on an unanswered question.
    air_most_economical: bool | None
    surface_mode: SurfaceMode | None  # the cheaper surface option, even if air is chosen
    travel_subsidy_aud: float | None = Field(default=None, ge=0)  # the chosen mode's cost
    costs: ModeCosts
    questions: list[NonEmptyStr] = Field(default_factory=list)
    rule_hits: list[RuleHit] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.chosen_mode == "air") != (self.air_basis is not None):
            raise ValueError("air has a basis exactly when air is the chosen mode")
        if self.air_basis == "economical" and self.air_most_economical is not True:
            raise ValueError("an economical air choice must be the most economical")
        return self


# --- Accommodation (§6.6.1, §7.3, §7.3.1) ------------------------------------


class AccommodationSubsidyInput(StrictModel):
    """One stay, and what decides how much of it PATS pays for."""

    model_config = ConfigDict(frozen=True)

    nights: int = Field(ge=1)
    travellers: int = Field(ge=1)  # 2 = a client and an authorised escort
    concession_card: bool = False
    first_night_waiver: EntitlementStatus = "not_entitled"
    commercial: bool = True  # §6.6.1: only commercial accommodation (with an ABN)
    staying_with_family_or_friends: bool = False  # §6.6.1: never subsidised
    nightly_cost_aud: float | None = Field(default=None, gt=0)  # the actual cost; None = unknown
    # §7.3.1: nights the treating specialist authorised beyond the automatic two.
    specialist_authorised_nights: int | None = Field(default=None, ge=1)


class AccommodationSubsidy(StrictModel):
    """What PATS would pay towards a stay. Read-only."""

    model_config = ConfigDict(frozen=True)

    status: Literal["calculated", "ask", "needs_review"]
    nights_subsidised: int | None = Field(default=None, ge=0)
    first_night_subsidised: bool | None
    per_night_cap_aud: float = Field(gt=0)  # $110 alone, $154 with an escort (§7.3)
    maximum_aud: float | None = Field(default=None, ge=0)  # nights subsidised x cap
    subsidy_aud: float | None = Field(default=None, ge=0)  # needs the actual cost
    questions: list[NonEmptyStr] = Field(default_factory=list)
    rule_hits: list[RuleHit] = Field(min_length=1)

    @model_validator(mode="after")
    def _unknowns_only_when_not_calculated(self) -> Self:
        if self.status != "calculated" and (self.maximum_aud is not None):
            raise ValueError("a stay that needs an answer or a review has no calculated maximum")
        if self.status == "ask" and not self.questions:
            raise ValueError("an 'ask' result must carry its question")
        return self


# --- Non-eligible travel (§6.5.8) ------------------------------------


class ModeFlags(StrictModel):
    """Which of the modes a client asked about PATS does not subsidise. Read-only."""

    model_config = ConfigDict(frozen=True)

    # The client's own words for each mode PATS does not subsidise.
    flagged: list[NonEmptyStr] = Field(default_factory=list)
    # Modes the tool recognises as neither excluded nor an ordinary eligible mode. The plan must
    # not call them fine.
    unrecognised: list[NonEmptyStr] = Field(default_factory=list)
    rule_hits: list[RuleHit] = Field(default_factory=list)


# --- Everything the plan needs from the calculator for one client ---------------------


class LogisticsAssessment(StrictModel):
    """The calculator's answer for an eligible client. Read-only."""

    model_config = ConfigDict(frozen=True)

    travellers: int = Field(ge=1)
    modes: ModeComparison
    # §7.3.1 with the §6.1.2 waiver: True/False once known, None while it depends on an answer.
    first_night_subsidised: bool | None
    # §6.5.8: modes the client asked about that PATS does not subsidise.
    ineligible_modes_flagged: list[NonEmptyStr] = Field(default_factory=list)
    # Modes the tool doesn't recognise as either eligible or ineligible. The plan must not say
    # they are fine.
    unrecognised_modes: list[NonEmptyStr] = Field(default_factory=list)
    questions: list[NonEmptyStr] = Field(default_factory=list)
    rule_hits: list[RuleHit] = Field(min_length=1)
