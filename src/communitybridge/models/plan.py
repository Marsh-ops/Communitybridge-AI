"""Logistics options and plan items: what the user actually sees in their support plan.

`LogisticsOption` is built by the Logistics Agent from the logistics tool and the simulated
transport, accommodation and calendar APIs. In this prototype every option comes from a
simulated service, so `simulated` must be True and the UI shows a SIMULATED badge.

`PlanItem` enforces the core rule in the data: a statement can only be "verified" if it has at
least one citation. Anything else must be marked "blocked" with a reason, which the UI shows.
"""

from datetime import date, datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from communitybridge.models.common import Citation, NonEmptyStr, StrictModel


class LogisticsOption(StrictModel):
    # A string, not an enum: the modes and their rules are in the rules data (PATS §7.2) and are
    # worked out by tools/logistics.py, so a guideline update changes the YAML, not this model.
    travel_mode: NonEmptyStr
    # Includes the escort when one is authorised (PATS §7.2.8), so the plan budgets for both.
    travellers: int = Field(ge=1)
    accommodation_id: str | None = None
    appointment_start: datetime
    citations: list[Citation] = Field(min_length=1)
    simulated: Literal[True]
    # What a booking request needs, and what the user is shown before approving it. All
    # optional so an option that only names the basics is still valid. They come from the
    # window matcher (tools/windows.py), which got them from the SIMULATED services.
    outbound_date: date | None = None
    return_date: date | None = None
    nights: int = Field(default=0, ge=0)
    outbound_route_id: str | None = None  # None when the client drives
    return_route_id: str | None = None
    accommodation_name: str | None = None
    accommodation_total_aud: float | None = Field(default=None, ge=0)  # the stay's actual cost
    accommodation_subsidy_aud: float | None = Field(default=None, ge=0)  # PATS pays up to (§7.3)
    travel_subsidy_aud: float | None = Field(default=None, ge=0)  # what PATS would pay (§7.2)
    notes: list[NonEmptyStr] = Field(default_factory=list)


class PlanItem(StrictModel):
    statement: NonEmptyStr
    citations: list[Citation] = Field(default_factory=list)
    status: Literal["verified", "blocked"]
    blocked_reason: str | None = None

    @model_validator(mode="after")
    def _verified_needs_citation_blocked_needs_reason(self) -> Self:
        if self.status == "verified" and not self.citations:
            raise ValueError("a verified plan item needs at least one citation")
        if self.status == "blocked" and not (self.blocked_reason and self.blocked_reason.strip()):
            raise ValueError("a blocked plan item needs a blocked_reason the user can read")
        return self
