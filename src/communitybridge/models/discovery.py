"""Results of the Service Discovery tool.

The Service Discovery Agent finds the nearest service for the client's specialty. Its rule in the
agent table is "every result carries its `source` and `last_verified`", so the
Verifier can later block a claim with no source or a stale one. These models make
that structural: a `ServiceMatch` cannot be built unless its `Citation` agrees with the service
record it came from.

The results are read-only. They describe which services *count* as nearest under §6.4.6; they
never say whether the client is eligible, which only the rules engine decides.
"""

from typing import Self

from pydantic import ConfigDict, Field, model_validator

from communitybridge.models.common import Citation, NonEmptyStr, StrictModel
from communitybridge.models.dataset import ServiceRecord


class ServiceMatch(StrictModel):
    """One service, how far it is from the client's home, and where the information came from."""

    model_config = ConfigDict(frozen=True)

    service: ServiceRecord
    distance_km: float = Field(ge=0)
    citation: Citation

    @model_validator(mode="after")
    def _citation_matches_the_record(self) -> Self:
        # The citation is what the Verifier checks for freshness, so it must be the record's own.
        if (
            self.citation.source != self.service.source
            or self.citation.last_verified != self.service.last_verified
        ):
            raise ValueError(
                "the citation must carry the service record's source and last_verified"
            )
        return self


class ServiceSearch(StrictModel):
    """The services for one specialty, nearest first.

    `nearest` holds every service tied for nearest. `others` holds the rest, nearest first, so
    the plan can offer an alternative if the client cannot use the nearest. Both are empty if no
    service in the dataset offers the specialty, which is a valid answer, not an error.
    """

    model_config = ConfigDict(frozen=True)

    origin_town_id: NonEmptyStr
    specialty: NonEmptyStr
    nearest: list[ServiceMatch] = Field(default_factory=list)
    others: list[ServiceMatch] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.others and not self.nearest:
            raise ValueError("there cannot be other services when there is no nearest one")
        if self.nearest:
            nearest_km = self.nearest[0].distance_km
            if any(match.distance_km != nearest_km for match in self.nearest):
                raise ValueError("every nearest service must be the same distance away")
            if any(match.distance_km <= nearest_km for match in self.others):
                raise ValueError("the other services must be farther than the nearest")
        order = [(match.distance_km, match.service.service_id) for match in self.others]
        if order != sorted(order):
            raise ValueError("the other services must be ordered nearest first")
        return self
