"""Schema for the 30 manually checked evaluation scenarios (data/scenarios/scenarios.yaml).

Each scenario is one user goal plus everything needed to mark the system's answer:
- `expected_fields`: what goal extraction should produce (success target: >= 90% field accuracy);
- `facts`: details the user or coordinator confirms that aren't in the goal text;
- `expected`: the eligibility outcome worked out by hand from the PATS PDF, the
  sections that must be cited, any out-of-scope notices and any entitlements
  (success target: 100% eligibility accuracy).

`extra="forbid"` on `ConfirmedFacts` means a misspelt fact name fails validation instead of being
silently ignored, which would otherwise make a scenario test the wrong thing.
"""

from datetime import date
from typing import Literal

from pydantic import Field

from communitybridge.models.common import NonEmptyStr, StrictModel
from communitybridge.models.engine import EligibilityOutcome
from communitybridge.models.facts import ConfirmedFacts
from communitybridge.models.request import GoalRequest

# Each category must cite at least one of its sections (the rule groups).
CATEGORY_SECTIONS: dict[str, set[str]] = {
    "core": {"6.5", "6.4.6", "6.1.2", "7.2.8", "7.2.5"},
    "travel_distance": {"6.5", "6.5.3"},  # §6.5.1 is out of scope
    "nearest_service": {"6.4.6", "8.1.1"},
    "apy_lands": {"6.1.1"},
    "aboriginal_engagement": {"6.1.2"},
    "escort": {"7.2.8", "7.2.9"},
    "air_vs_surface": {"7.2.5", "7.2.6"},
    "non_eligible_travel": {"6.5.8"},
    "specialty_not_accepted": {"9.2"},
    "ineligible_appointment": {"6.4.5"},
}
Category = Literal[
    "core",
    "travel_distance",
    "nearest_service",
    "apy_lands",
    "aboriginal_engagement",
    "escort",
    "air_vs_surface",
    "non_eligible_travel",
    "specialty_not_accepted",
    "ineligible_appointment",
]
Outcome = EligibilityOutcome

# The scenarios' "facts" are the same details the engine gets in the app (models/facts.py), so
# the engine is tested with exactly its real input. The old name is kept for existing imports.
ScenarioFacts = ConfirmedFacts


class ExpectedEntitlements(StrictModel):
    """Only the entitlements a scenario is about are set; None means "not checked here"."""

    escort: bool | None = None
    advance: bool | None = None
    first_night_subsidised: bool | None = None
    air_most_economical: bool | None = None
    surface_mode: Literal["private_vehicle", "public_transport"] | None = None
    ineligible_modes_flagged: list[str] | None = None


class ExpectedResult(StrictModel):
    outcome: Outcome
    must_cite: list[str] = Field(min_length=1)
    # Out-of-scope sections the plan must flag as "not assessed", e.g. §6.5.1.
    out_of_scope_notices: list[str] = Field(default_factory=list)
    entitlements: ExpectedEntitlements = Field(default_factory=ExpectedEntitlements)
    working: NonEmptyStr  # the hand calculation, citing sections and interpretations


class Scenario(StrictModel):
    id: str = Field(pattern=r"^SC-\d{2}$")
    category: Category
    title: NonEmptyStr
    goal: NonEmptyStr
    expected_fields: GoalRequest
    facts: ConfirmedFacts = Field(default_factory=ConfirmedFacts)
    destination_service_id: NonEmptyStr
    expected: ExpectedResult
    synthetic: Literal[True]


class ScenarioSet(StrictModel):
    reference_date: date  # "next month" etc. are resolved against this date
    rules_version: NonEmptyStr
    decision_records: list[str] = Field(default_factory=list)
    scenarios: list[Scenario]
