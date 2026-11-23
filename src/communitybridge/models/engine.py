"""The rules engine's complete answer for one client and one chosen service.

`Decision` is the contract in the project design: `eligible`, `rule_hits` and `rules_version`. It has
one yes/no flag, but PATS eligibility has three possible outcomes: eligible, not eligible, and
"needs review" (a PATS authorised officer must decide,). The flag alone can't tell the
last two apart, because each rule reports honestly on its own test. For example, §6.4.6 says
`fail` for a trip to a farther service even when a GP-documented reason then makes the trip
acceptable. So `EligibilityResult` carries the combined `outcome` next to the `Decision`.

It also carries what the plan needs beyond the yes/no: the "not assessed" notices for the sections
this prototype leaves out of scope (§6.5.1, §6.4.7), the nearest services to offer instead, and the
entitlements (advance, escort). Like `Decision`, it is read-only, so no agent can change it.
"""

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from communitybridge.models.common import NonEmptyStr, StrictModel
from communitybridge.models.distance import OutOfScopeNotice
from communitybridge.models.eligibility import Decision
from communitybridge.models.entitlements import EntitlementAssessment

EligibilityOutcome = Literal["eligible", "not_eligible", "needs_review"]


class EligibilityResult(StrictModel):
    """Everything the rules engine concluded. Read-only input for every agent after it."""

    model_config = ConfigDict(frozen=True)

    decision: Decision
    outcome: EligibilityOutcome
    # The services tied for nearest, so the plan can offer one when the chosen service is not.
    nearest_service_ids: list[NonEmptyStr] = Field(min_length=1)
    notices: list[OutOfScopeNotice] = Field(default_factory=list)
    # Only worked out for an eligible client: an entitlement means nothing to someone who is not.
    entitlements: EntitlementAssessment | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.decision.eligible != (self.outcome == "eligible"):
            raise ValueError("decision.eligible must be true exactly when the outcome is eligible")
        if (self.entitlements is not None) != (self.outcome == "eligible"):
            raise ValueError("entitlements are given exactly when the outcome is eligible")
        return self
