"""Input and output models for the entitlement rules.

Entitlements are what a client may *receive* once they are eligible: the Aboriginal Engagement
Strategy benefits (§6.1.2), an advance on travel subsidies (§6.5.5) and an escort (§7.2.8,
§7.2.9). Not having one is not a failure: the client stays eligible for everything else. So the
result says `entitled`, `not_entitled` or `ask`, and its rule hits use `pass` or `not_applicable`,
never `fail`.

`ask` follows this rule: where an entitlement depends on whether the client is
Aboriginal or Torres Strait Islander and they have not said, the plan asks the question and shows
the entitlement as "depends on your answer". It does not block the decision (only the §6.1.1 APY
Lands exclusion does that).
"""

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from communitybridge.models.common import NonEmptyStr, StrictModel
from communitybridge.models.eligibility import RuleHit
from communitybridge.models.request import IndigenousStatus

# The clinical criteria in the §7.2.9 table that need the treating specialist's approval. The two
# automatic ones ("Client is a child", "Aboriginal and/or Torres Strait Islander people") are
# decided from age and identity, so they are not values here.
EscortCriterion = Literal[
    "Impairment",
    "Active role of carer",
    "Necessary assistance",
    "As an alternative to air travel",
]

EntitlementStatus = Literal["entitled", "not_entitled", "ask"]
EscortStatus = Literal["not_requested", "entitled", "not_entitled", "ask"]


class EntitlementInput(StrictModel):
    """What the entitlement rules need to know about one client."""

    model_config = ConfigDict(frozen=True)

    indigenous_status: IndigenousStatus = "not_stated"
    # Whether the client asked for an escort. The rules are only applied (and cited) if so.
    escort_requested: bool = False
    client_age: int = Field(default=45, ge=0)
    # §7.2.9: the clinical criterion the treating specialist has approved an escort under.
    escort_criterion: EscortCriterion | None = None
    escort_specialist_approved: bool = False
    # §6.5.5: the three criteria for an advance (a client who is not covered by §6.1.2).
    concession_card: bool = False
    severe_financial_hardship: bool = False
    notice_business_days: int | None = Field(default=None, ge=0)  # None = not stated


class EntitlementAssessment(StrictModel):
    """Result of the entitlement rules. Read-only, like the engine's Decision."""

    model_config = ConfigDict(frozen=True)

    escort: EscortStatus
    advance: EntitlementStatus
    # §6.1.2: the concession card is not needed for the first night's accommodation subsidy.
    first_night_concession_waived: EntitlementStatus
    # Questions the plan must put to the client, one for each "ask" above.
    questions: list[NonEmptyStr] = Field(default_factory=list)
    rule_hits: list[RuleHit] = Field(min_length=1)

    @model_validator(mode="after")
    def _questions_match_asks(self) -> Self:
        asking = "ask" in (self.escort, self.advance, self.first_night_concession_waived)
        if asking != bool(self.questions):
            raise ValueError("there must be questions exactly when an entitlement is 'ask'")
        # An entitlement that is not granted is information, not a failure (see module docstring).
        if any(hit.outcome in ("fail", "needs_review") for hit in self.rule_hits):
            raise ValueError("entitlement hits are pass or not_applicable, never fail or review")
        return self
