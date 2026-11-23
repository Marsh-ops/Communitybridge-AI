"""Records of the synthetic, community-administered dataset (towns, distances, services, stays).

These models validate every row before it reaches SQLite. Like `Service`, each has
`synthetic: Literal[True]`, so a record that isn't synthetic is rejected at load time.
"""

from datetime import date
from typing import Literal, Self

from pydantic import Field, model_validator

from communitybridge.models.common import NonEmptyStr, StrictModel
from communitybridge.models.service import Service


class Town(StrictModel):
    town_id: NonEmptyStr
    name: NonEmptyStr
    region: NonEmptyStr
    apy_lands: bool  # PATS §6.1.1
    synthetic: Literal[True]


class Distance(StrictModel):
    """One-way road distance. Synthetic stand-in for the PATS distance calculator (§6.5.2)."""

    from_town_id: NonEmptyStr
    to_town_id: NonEmptyStr
    road_km: float = Field(ge=0)
    source: NonEmptyStr
    synthetic: Literal[True]


class ServiceRecord(Service):
    """A Service as stored in the dataset file, plus the dates a visiting specialist attends."""

    visit_dates: list[date] = Field(default_factory=list)

    @model_validator(mode="after")
    def _visit_dates_only_for_visiting(self) -> Self:
        # A visiting service with no visit dates could never be "nearest" (§6.4.6), and a
        # resident or telehealth service with visit dates would be contradictory.
        if (self.mode == "visiting") != bool(self.visit_dates):
            raise ValueError("visit_dates are required for visiting services and only for them")
        return self


class Accommodation(StrictModel):
    accommodation_id: NonEmptyStr
    name: NonEmptyStr
    town: NonEmptyStr
    commercial: bool  # PATS §6.6.1: only commercial accommodation (with an ABN) is subsidised
    source: NonEmptyStr
    last_verified: date
    synthetic: Literal[True]
