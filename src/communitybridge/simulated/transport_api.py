"""SIMULATED transport service: buses and flights between country towns and Adelaide.

Serves the Logistics Agent and the logistics calculator. Every response is a
`SimulatedResponse`, so it carries `simulated: true` and the label SIMULATED. The
timetable and fares are invented in `data/simulated/transport.yaml`; nothing here talks to a real
operator.

Bus fares are one way per person and air fares are return per person, as PATS §7.2.5 compares
them. A route is found in either direction and is assumed to run the same days both ways, which
is a simplification of this simulation, not a claim about real services.
"""

from collections.abc import Sequence
from datetime import date

from communitybridge.models.simulated import (
    BookingRequest,
    FareSheet,
    RouteFare,
    RouteRecord,
    TransportOption,
    TransportSearch,
)
from communitybridge.simulated.common import (
    InvalidRequestError,
    NotAvailableError,
    SimulatedActionLog,
    require_approval,
    weekday_full,
    weekday_short,
)


class SimulatedTransportApi:
    def __init__(
        self,
        routes: Sequence[RouteRecord],
        log: SimulatedActionLog,
        last_verified: date | None = None,
    ) -> None:
        self._routes = {route.route_id: route for route in routes}
        self._log = log
        # The date the fixture file says it was checked; a plan option cites it.
        self.last_verified = last_verified

    def route(self, route_id: str) -> RouteRecord:
        """One route's record, so a plan option can cite exactly the route it uses."""
        route = self._routes.get(route_id)
        if route is None:
            raise NotAvailableError(f"route {route_id} is not in the simulated timetable")
        return route

    def _option(self, route: RouteRecord, travel_date: date) -> TransportOption:
        reason = None
        if weekday_short(travel_date) not in route.days:
            reason = f"no service on {weekday_full(travel_date)}s (SIMULATED timetable)"
        elif travel_date in route.unavailable_dates:
            reason = "no seats on that date (SIMULATED)"
        return TransportOption(
            route_id=route.route_id,
            mode=route.mode,
            from_town=route.from_town,
            to_town=route.to_town,
            fare_aud=route.fare_aud,
            fare_type=route.fare_type,
            depart=route.depart,
            arrive=route.arrive,
            travel_date=travel_date,
            available=reason is None,
            reason=reason,
        )

    def search(self, from_town: str, to_town: str, travel_date: date) -> TransportSearch:
        """Every route between the two towns (either way round), with whether it runs that day."""
        pair = {from_town, to_town}
        routes = [r for r in self._routes.values() if {r.from_town, r.to_town} == pair]
        return TransportSearch(
            from_town=from_town,
            to_town=to_town,
            travel_date=travel_date,
            options=[
                self._option(r, travel_date) for r in sorted(routes, key=lambda r: r.route_id)
            ],
        )

    def fares(self, from_town: str, to_town: str) -> FareSheet:
        """The price list between two towns, whatever the day. Used to compare costs (§7.2.5)."""
        pair = {from_town, to_town}
        routes = [r for r in self._routes.values() if {r.from_town, r.to_town} == pair]
        return FareSheet(
            from_town=from_town,
            to_town=to_town,
            routes=[
                RouteFare(
                    route_id=r.route_id,
                    mode=r.mode,
                    from_town=r.from_town,
                    to_town=r.to_town,
                    fare_aud=r.fare_aud,
                    fare_type=r.fare_type,
                )
                for r in sorted(routes, key=lambda r: r.route_id)
            ],
        )

    def request_booking(
        self, route_id: str, travel_date: date, travellers: int, approval_id: str
    ) -> BookingRequest:
        """Make a SIMULATED request for seats. Needs an approval id; the fare is per traveller."""
        approval = require_approval(approval_id)
        if travellers < 1:
            raise InvalidRequestError("a booking needs at least one traveller")
        route = self._routes.get(route_id)
        if route is None:
            raise NotAvailableError(f"route {route_id} is not in the simulated timetable")
        option = self._option(route, travel_date)
        if not option.available:
            raise NotAvailableError(f"route {route_id} on {travel_date}: {option.reason}")
        return self._log.record(
            "transport",
            f"{route.mode} {route.route_id}, {route.from_town} to {route.to_town}, "
            f"{travel_date}, {travellers} traveller(s)",
            route.fare_aud * travellers,
            approval,
        )
