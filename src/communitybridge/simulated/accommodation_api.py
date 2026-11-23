"""SIMULATED accommodation service: prices and availability for a stay.

Serves the Logistics Agent and the logistics calculator. Which places exist, in which town
and whether they are commercial (PATS §6.6.1) come from the curated accommodation dataset. The
nightly rates and the nights with no room are invented in `data/simulated/accommodation.yaml`.
Every response is a `SimulatedResponse`, so it carries `simulated: true` and the label SIMULATED. A place with no price fixture is left out of a search, never given an invented
price.

A stay of N nights starting on `check_in` uses the nights `check_in`, `check_in + 1`, and so on;
the day the client leaves is not a night.
"""

from collections.abc import Sequence
from datetime import date, timedelta

from communitybridge.models.dataset import Accommodation
from communitybridge.models.simulated import (
    AccommodationOption,
    AccommodationSearch,
    AvailabilityRecord,
    BookingRequest,
)
from communitybridge.simulated.common import (
    InvalidRequestError,
    NotAvailableError,
    SimulatedActionLog,
    require_approval,
)


class SimulatedAccommodationApi:
    def __init__(
        self,
        accommodation: Sequence[Accommodation],
        availability: Sequence[AvailabilityRecord],
        log: SimulatedActionLog,
    ) -> None:
        prices = {record.accommodation_id: record for record in availability}
        # Only places that have a price fixture can be offered.
        self._places = {
            place.accommodation_id: (place, prices[place.accommodation_id])
            for place in accommodation
            if place.accommodation_id in prices
        }
        self._log = log

    def _option(
        self, place: Accommodation, prices: AvailabilityRecord, check_in: date, nights: int
    ) -> AccommodationOption:
        stay = [check_in + timedelta(days=i) for i in range(nights)]
        full = [night for night in stay if night in prices.unavailable_dates]
        reason = (
            f"no room on {', '.join(str(night) for night in full)} (SIMULATED)" if full else None
        )
        return AccommodationOption(
            accommodation_id=place.accommodation_id,
            name=place.name,
            town=place.town,
            commercial=place.commercial,
            nightly_rate_aud=prices.nightly_rate_aud,
            check_in=check_in,
            nights=nights,
            total_aud=prices.nightly_rate_aud * nights,
            available=reason is None,
            reason=reason,
        )

    def place(self, accommodation_id: str) -> Accommodation:
        """One place's dataset record, so a booking can find its town."""
        if accommodation_id not in self._places:
            raise NotAvailableError(f"{accommodation_id} is not in the simulated accommodation")
        return self._places[accommodation_id][0]

    def search(self, town: str, check_in: date, nights: int) -> AccommodationSearch:
        """Every place in the town with a price, and whether every night of the stay has a room."""
        _check_nights(nights)
        options = [
            self._option(place, prices, check_in, nights)
            for place, prices in sorted(self._places.values(), key=lambda p: p[0].accommodation_id)
            if place.town == town
        ]
        return AccommodationSearch(town=town, check_in=check_in, nights=nights, options=options)

    def request_booking(
        self, accommodation_id: str, check_in: date, nights: int, approval_id: str
    ) -> BookingRequest:
        """Make a SIMULATED request for a stay. Needs an approval id; the total is rate x nights."""
        approval = require_approval(approval_id)
        _check_nights(nights)
        if accommodation_id not in self._places:
            raise NotAvailableError(f"{accommodation_id} is not in the simulated accommodation")
        place, prices = self._places[accommodation_id]
        option = self._option(place, prices, check_in, nights)
        if not option.available:
            raise NotAvailableError(f"{accommodation_id} from {check_in}: {option.reason}")
        return self._log.record(
            "accommodation",
            f"{place.name} ({accommodation_id}), {nights} night(s) from {check_in}",
            option.total_aud,
            approval,
        )


def _check_nights(nights: int) -> None:
    if nights < 1:
        raise InvalidRequestError("a stay needs at least one night")
