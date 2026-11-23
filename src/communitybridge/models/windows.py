"""Models for appointment-window matching.

The project plan's fourth in-scope step is to suggest "transport, accommodation and appointment
windows that work together". A *window* here is one appointment slot plus the travel and stay
around it. The matcher checks each slot against the SIMULATED transport and accommodation, and
reports both what works and, for what doesn't, the reasons. The reasons matter: they are what the
Coordinator's re-planning follows when no window works (the design diagram,).

Everything here came from a SIMULATED service, so every result is a `SimulatedResponse` and
carries `simulated: true` and the SIMULATED label. A window is only a suggestion; it
books nothing (a booking needs an approval, and is made elsewhere).
"""

from datetime import date
from typing import Self

from pydantic import ConfigDict, Field, model_validator

from communitybridge.models.common import NonEmptyStr, StrictModel
from communitybridge.models.entitlements import EntitlementStatus
from communitybridge.models.logistics import AccommodationSubsidy, TravelMode
from communitybridge.models.simulated import AppointmentSlot, SimulatedResponse


class WindowRequest(StrictModel):
    """What the matcher needs to know about the journey. Built from the logistics assessment."""

    model_config = ConfigDict(frozen=True)

    destination_town_id: NonEmptyStr
    travellers: int = Field(ge=1)
    # The mode the logistics calculator chose (§7.2). None means it isn't decided yet.
    travel_mode: TravelMode | None
    # What PATS would pay towards that travel (the chosen mode's cost), shown with each window.
    travel_cost_aud: float | None = Field(default=None, ge=0)
    origin_town_id: NonEmptyStr
    concession_card: bool = False
    first_night_waiver: EntitlementStatus = "not_entitled"
    staying_with_family_or_friends: bool = False  # §6.6.1: then no accommodation is booked


class WindowOption(SimulatedResponse):
    """One slot that works, with the travel and stay around it."""

    appointment: AppointmentSlot
    # None for a telehealth appointment, which needs no travel.
    travel_mode: TravelMode | None
    outbound_date: date | None
    return_date: date | None
    nights: int = Field(ge=0)
    outbound_route_id: str | None = None  # None when the client drives, or for telehealth
    return_route_id: str | None = None
    accommodation_id: str | None = None  # None with family or friends, or for telehealth
    accommodation_name: str | None = None
    accommodation_total_aud: float | None = Field(default=None, ge=0)  # the actual cost of the stay
    accommodation: AccommodationSubsidy | None = None  # what PATS would pay towards it (§7.3)
    travel_cost_aud: float | None = Field(default=None, ge=0)
    notes: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _telehealth_needs_no_travel(self) -> Self:
        if self.appointment.mode == "telehealth":
            if (
                self.travel_mode is not None
                or self.nights
                or self.outbound_date
                or self.accommodation_id
            ):
                raise ValueError("a telehealth appointment needs no travel or accommodation")
        elif self.outbound_date is None or self.return_date is None:
            raise ValueError("an in-person appointment needs an outbound and a return date")
        return self


class RejectedWindow(SimulatedResponse):
    """A slot that does not work, and every reason why. The re-planning follows these."""

    appointment: AppointmentSlot
    reasons: list[NonEmptyStr] = Field(min_length=1)


class WindowMatches(SimulatedResponse):
    """Every slot in the client's dates, sorted into what works and what doesn't."""

    options: list[WindowOption] = Field(default_factory=list)
    rejected: list[RejectedWindow] = Field(default_factory=list)
