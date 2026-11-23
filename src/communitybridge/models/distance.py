"""Input and output models for the distance tool.

The tool is deterministic, so it takes a validated input model and returns a validated, read-only
result (the design approach: language models interpret and explain, deterministic tools decide). The
result uses the same `RuleHit` as the rules engine, so the Eligibility Agent can cite each
section without a second format.

`status` has three values on purpose:
- `met`: the distance requirement is satisfied (§6.5, or the §6.5.3 band);
- `not_met`: it is not, so the plan must show the §6.5.1 "not assessed" notice;
- `needs_review`: a fact needed for the §6.5.3 band is missing. The tool does not guess; a PATS
  authorised officer decides (§8.3,).
"""

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from communitybridge.models.common import SECTION_PATTERN, NonEmptyStr, StrictModel
from communitybridge.models.eligibility import RuleHit

DistanceStatus = Literal["met", "not_met", "needs_review"]


class TravelDistanceInput(StrictModel):
    """What the distance rules need to know about one trip."""

    model_config = ConfigDict(frozen=True)

    # One way, from the client's permanent residence to the treatment location (§6.5.2).
    distance_km: float = Field(ge=0)
    # The treatment performed; matched against the §6.5.3 list of treatments that leave a client
    # lethargic. None means "not known yet".
    treatment: str | None = None
    # The alternate route's length (from Google Maps, §6.5.3). None means "not known yet".
    alternate_route_km: float | None = Field(default=None, ge=0)


class OutOfScopeNotice(StrictModel):
    """A "this prototype doesn't assess this" notice for a section outside the project scope."""

    model_config = ConfigDict(frozen=True)

    section: str = Field(pattern=SECTION_PATTERN)
    title: NonEmptyStr
    message: NonEmptyStr
    source_page: int = Field(ge=1)  # kept so the notice itself carries a citation


class TravelDistanceAssessment(StrictModel):
    """Result of the travel-distance rules. Read-only, like the engine's Decision."""

    model_config = ConfigDict(frozen=True)

    distance_km: float = Field(ge=0)
    status: DistanceStatus
    rule_hits: list[RuleHit] = Field(min_length=1)
    notices: list[OutOfScopeNotice] = Field(default_factory=list)

    @model_validator(mode="after")
    def _notices_only_when_not_met(self) -> Self:
        # the notice exists to stop a client being told a bare "not eligible". It is not
        # shown when the requirement is met, or while it is still unknown (needs_review).
        if self.notices and self.status != "not_met":
            raise ValueError("out-of-scope notices are only shown when the requirement is not met")
        return self
