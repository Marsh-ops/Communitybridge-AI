"""Models for the simulated transport, accommodation and calendar services.

The core rule: simulated APIs are clearly labelled. Every simulated response carries
`"simulated": true`, and the UI shows a visible SIMULATED badge. Both are made structural here:

- Every response type inherits `SimulatedResponse`, whose `simulated` field is `Literal[True]` and
  whose `label` is the badge text. A response cannot be built with `simulated=False`, and it
  cannot leave the label out, so a labelling slip is a validation error, not a quiet bug.
- Every fixture row (`RouteRecord`, `AvailabilityRecord`, `SlotRecord`) must itself say
  `simulated: true` and `synthetic: true`, so an unmarked row is refused on load.

Nothing here is real: no timetable, fare, room or appointment exists outside the fixtures in
`data/simulated/`, and a "booking" is only a request that the prototype records.
"""

from datetime import date, datetime
from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from communitybridge.models.common import NonEmptyStr, StrictModel

Weekday = Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TransportMode = Literal["bus", "air"]
# Bus fares are one way per person; air fares are return per person (PATS §7.2.5 compares them so).
FareType = Literal["one_way", "return"]
SlotMode = Literal["in_person", "telehealth"]
# "fallback_only" is the referred specialist's own telehealth slot, offered only when travel
# can't be arranged. "listed_service" belongs to a telehealth service in the dataset.
OfferedAs = Literal["listed_service", "fallback_only"]
TIME_PATTERN = r"^([01]\d|2[0-3]):[0-5]\d$"


class SimulatedResponse(StrictModel):
    """Base of every response from a simulated service. Read-only, and always labelled."""

    model_config = ConfigDict(frozen=True)

    simulated: Literal[True] = True
    label: Literal["SIMULATED"] = "SIMULATED"  # the text of the badge the UI shows


# --- Fixture rows (data/simulated/*.yaml) ------------------------------------


class RouteRecord(StrictModel):
    route_id: NonEmptyStr
    mode: TransportMode
    from_town: NonEmptyStr
    to_town: NonEmptyStr
    fare_aud: float = Field(gt=0)
    fare_type: FareType
    days: list[Weekday] = Field(min_length=1)
    depart: str = Field(pattern=TIME_PATTERN)
    arrive: str = Field(pattern=TIME_PATTERN)
    unavailable_dates: list[date] = Field(default_factory=list)
    simulated: Literal[True]
    synthetic: Literal[True]


class AvailabilityRecord(StrictModel):
    accommodation_id: NonEmptyStr
    nightly_rate_aud: float = Field(gt=0)
    unavailable_dates: list[date] = Field(default_factory=list)
    simulated: Literal[True]
    synthetic: Literal[True]


class SlotRecord(StrictModel):
    mode: SlotMode
    service_id: NonEmptyStr
    starts: list[datetime] = Field(min_length=1)
    telehealth_host_town: NonEmptyStr | None = None
    offered_as: OfferedAs | None = None
    simulated: Literal[True]
    synthetic: Literal[True]

    @model_validator(mode="after")
    def _telehealth_rows_say_where_and_how(self) -> Self:
        telehealth = self.mode == "telehealth"
        if telehealth != (self.telehealth_host_town is not None and self.offered_as is not None):
            raise ValueError(
                "telehealth slots need a host town and offered_as, and in-person slots must not "
                "have them"
            )
        return self


# --- Transport ------------------------------------


class TransportOption(SimulatedResponse):
    route_id: NonEmptyStr
    mode: TransportMode
    from_town: NonEmptyStr
    to_town: NonEmptyStr
    fare_aud: float = Field(gt=0)
    fare_type: FareType
    depart: NonEmptyStr
    arrive: NonEmptyStr
    travel_date: date
    available: bool
    reason: str | None = None  # why it is not available, in plain words

    @model_validator(mode="after")
    def _reason_only_when_unavailable(self) -> Self:
        if self.available == (self.reason is not None):
            raise ValueError("an unavailable option needs a reason, and an available one has none")
        return self


class TransportSearch(SimulatedResponse):
    from_town: NonEmptyStr
    to_town: NonEmptyStr
    travel_date: date
    options: list[TransportOption] = Field(default_factory=list)


class RouteFare(SimulatedResponse):
    """A route's fare, with no date: the price list, not the timetable for a day."""

    route_id: NonEmptyStr
    mode: TransportMode
    from_town: NonEmptyStr
    to_town: NonEmptyStr
    fare_aud: float = Field(gt=0)
    fare_type: FareType


class FareSheet(SimulatedResponse):
    """Every fare between two towns. The logistics calculator compares costs from this."""

    from_town: NonEmptyStr
    to_town: NonEmptyStr
    routes: list[RouteFare] = Field(default_factory=list)


# --- Accommodation ------------------------------------


class AccommodationOption(SimulatedResponse):
    accommodation_id: NonEmptyStr
    name: NonEmptyStr
    town: NonEmptyStr
    commercial: bool  # PATS §6.6.1: only commercial accommodation (with an ABN) is subsidised
    nightly_rate_aud: float = Field(gt=0)
    check_in: date
    nights: int = Field(ge=1)
    total_aud: float = Field(gt=0)
    available: bool
    reason: str | None = None

    @model_validator(mode="after")
    def _reason_only_when_unavailable(self) -> Self:
        if self.available == (self.reason is not None):
            raise ValueError("an unavailable option needs a reason, and an available one has none")
        return self


class AccommodationSearch(SimulatedResponse):
    town: NonEmptyStr
    check_in: date
    nights: int = Field(ge=1)
    options: list[AccommodationOption] = Field(default_factory=list)


# --- Calendar ------------------------------------


class AppointmentSlot(SimulatedResponse):
    service_id: NonEmptyStr
    starts: datetime
    mode: SlotMode
    telehealth_host_town: NonEmptyStr | None = None
    offered_as: OfferedAs | None = None


class CalendarSearch(SimulatedResponse):
    service_id: NonEmptyStr
    earliest: date
    latest: date
    slots: list[AppointmentSlot] = Field(default_factory=list)


# --- A simulated booking request ------------------------------------


class BookingRequest(SimulatedResponse):
    """What a simulated "booking" returns: a request that was recorded, nothing more.

    `approval_id` is the recorded approval event it was made under. The simulated services refuse
    to make a request without one (defence in depth for "zero unapproved external actions").
    """

    kind: Literal["transport", "accommodation", "appointment"]
    reference: str = Field(pattern=r"^SIM-(TRN|ACC|APT)-\d{4}$")
    # "cancelled" is what withdrawing consent does to the SIMULATED hold.
    status: Literal["requested", "cancelled"] = "requested"
    description: NonEmptyStr
    total_aud: float | None = Field(default=None, ge=0)
    approval_id: NonEmptyStr
