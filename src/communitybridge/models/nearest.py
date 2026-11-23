"""Input and output models for the nearest-service rules.

PATS pays travel only to the client's nearest medical specialist service, and resident, visiting
and telehealth services all count. The only ways past a nearer service are the
reasons the PDF lists, so the reason is a closed list (`BypassReason`): a free-text reason such as
"family" could not be matched against what the PDF says is, and is not, enough.

Like the distance tool and the gates, the tool takes one validated input and returns one
read-only result, so no agent can change it afterwards (the design approach).
"""

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from communitybridge.models.common import NonEmptyStr, StrictModel
from communitybridge.models.distance import OutOfScopeNotice
from communitybridge.models.eligibility import RuleHit

# The three criteria the PDF accepts (§6.4.6, p. 9) ...
# ... the reasons it says are "not sufficient grounds" (p. 8) ...
# ... and the two it says are not "legitimate reasons" (p. 8-9). Each name is the PDF's own
# wording with underscores, which is how the tool finds it in the rules data.
BypassReason = Literal[
    "local_timeframe_clinically_unacceptable",
    "clinical_risks_cannot_be_managed_regionally",
    "cannot_be_treated_in_sa",
    "family_support",
    "continuity_of_care",
    "free_accommodation",
    "preferred_place_of_treatment",
    "personal_convenience",
    "special_interest_in_a_particular_area_of_specialty",
    "bypassing_waiting_lists_for_non_urgent_treatment",
]

NearestStatus = Literal["met", "not_met", "needs_review"]


class NearestServiceInput(StrictModel):
    """What the nearest-service rules need to know about one client and one chosen service."""

    model_config = ConfigDict(frozen=True)

    origin_town_id: NonEmptyStr
    destination_service_id: NonEmptyStr
    # What the client or their GP gives as the reason for travelling past a nearer service.
    bypass_reason: BypassReason | None = None
    # The client's GP "must provide a valid medical reason" (§6.4.6 p. 8).
    gp_reason_documented: bool = False
    # False means the client cannot be seen locally within a clinically acceptable timeframe.
    local_timeframe_clinically_acceptable: bool = True
    # The visiting specialist's primary rooms, if the client is going there (§6.4.6 p. 8).
    visiting_primary_rooms_service_id: NonEmptyStr | None = None
    # "cannot be treated in South Australia" referrals are "only accepted from a SA Health
    # tertiary health site" (§6.4.6 p. 9). None means not stated.
    referred_from_tertiary_site: bool | None = None


class NearestServiceAssessment(StrictModel):
    """Result of the nearest-service rules. Read-only, like the engine's Decision."""

    model_config = ConfigDict(frozen=True)

    status: NearestStatus
    # Every service tied for nearest, so the plan can offer the client the one they may use.
    nearest_service_ids: list[NonEmptyStr] = Field(min_length=1)
    nearest_km: float = Field(ge=0)
    destination_km: float = Field(ge=0)
    rule_hits: list[RuleHit] = Field(min_length=1)
    notices: list[OutOfScopeNotice] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.destination_km < self.nearest_km:
            raise ValueError("the destination cannot be nearer than the nearest service")
        # The notice only exists to stop a bare "not eligible".
        if self.notices and self.status != "not_met":
            raise ValueError("out-of-scope notices are only shown when the requirement is not met")
        return self
