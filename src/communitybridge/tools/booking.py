"""Booking tool: SIMULATED booking requests for one chosen option, only under an approval.

The Logistics Agent's second tool (the project design: "Yes, before any (simulated) booking request").
It turns one chosen `LogisticsOption` into the requests the SIMULATED services record: the
appointment, the transport (none for a client who drives, none for telehealth) and the stay. Nothing
real is booked, paid for or sent anywhere, and every request is labelled SIMULATED.

Two safeguards, because "zero unapproved external actions" is a success target and one safeguard
is not enough to promise it:
- **The approval id is checked first.** Without the id of a recorded approval event the tool
  refuses before it touches anything, and the simulated services refuse again on their own
  (`simulated/common.py`). The graph step that calls this tool also refuses unless the trace holds
  the approval (`agents/logistics.py`).
- **All or nothing.** Every part is checked for availability before the first request is made, so
  a stay that is full cannot leave a half-made booking behind: either every request is recorded,
  or none is.

Air fares are return fares, so one air request covers both directions. A bus fare is one way, so
a bus journey is two requests, out and back.

The caller says which *kinds* of request to make (`kinds`): the user approves the
release of data for each action separately (Gate 4), so a request is made only for the kinds they
approved, and a kind that was not asked for is neither checked nor made. Which kinds an option
needs is decided here, in `kinds_for_option`, and the Consent step uses the same function to know
which actions to ask about, so the two cannot disagree.
"""

from collections.abc import Sequence
from datetime import date

from communitybridge.models.plan import LogisticsOption
from communitybridge.models.simulated import BookingRequest
from communitybridge.simulated.common import (
    InvalidRequestError,
    NotAvailableError,
    require_approval,
)
from communitybridge.simulated.services import SimulatedServices


def _transport_legs(option: LogisticsOption) -> list[tuple[str, date]]:
    """The (route, day) of each transport request the option needs. Empty if the client drives or
    has a telehealth appointment."""
    legs = []
    if option.outbound_route_id and option.outbound_date:
        legs.append((option.outbound_route_id, option.outbound_date))
    # A return fare already covers the way back; a one-way fare needs its own request.
    if option.return_route_id and option.return_date and option.travel_mode != "air":
        legs.append((option.return_route_id, option.return_date))
    return legs


def _needs_stay(option: LogisticsOption) -> bool:
    return bool(option.accommodation_id and option.outbound_date and option.nights)


def kinds_for_option(option: LogisticsOption) -> list[str]:
    """The kinds of request the option needs, in the order they are made: the appointment always,
    the transport unless the client drives or it is telehealth, and the stay if there is one."""
    kinds = ["appointment"]
    if _transport_legs(option):
        kinds.append("transport")
    if _needs_stay(option):
        kinds.append("accommodation")
    return kinds


def _check_transport(legs: list[tuple[str, date]], simulated: SimulatedServices) -> None:
    """Each leg must have a seat. Raises `NotAvailableError` for the first that has none."""
    for route_id, day in legs:
        route = simulated.transport.route(route_id)
        found = simulated.transport.search(route.from_town, route.to_town, day)
        # The route exists (route() would have refused), so it is in its own towns' search.
        chosen = next(o for o in found.options if o.route_id == route_id)
        if not chosen.available:
            raise NotAvailableError(f"route {route_id} on {day}: {chosen.reason}")


def request_journey_booking(
    option: LogisticsOption,
    service_id: str,
    approval_id: str,
    simulated: SimulatedServices,
    kinds: Sequence[str] | None = None,
) -> list[BookingRequest]:
    """Make the SIMULATED requests for one option. Needs an approval id; all or nothing.

    `kinds` are the kinds of request to make (default: every kind the option needs). Raises
    `ApprovalRequiredError` without an approval id, `InvalidRequestError` for an empty, repeated or
    unneeded kind, and `NotAvailableError` if any part of what was asked for is no longer
    available. In every case no request has been made.
    """
    approval = require_approval(approval_id)
    needed = kinds_for_option(option)
    asked = list(needed if kinds is None else kinds)
    if not asked or len(asked) != len(set(asked)) or not set(asked) <= set(needed):
        raise InvalidRequestError(
            f"the kinds asked for ({', '.join(asked) or 'none'}) are not a non-empty, "
            f"non-repeating part of what this option needs ({', '.join(needed)})"
        )

    # Check everything asked for first, so a failure leaves nothing recorded.
    if "appointment" in asked:
        slot_day = option.appointment_start.date()
        slots = simulated.calendar.search(service_id, slot_day, slot_day, include_fallback=True)
        if option.appointment_start not in {slot.starts for slot in slots.slots}:
            raise NotAvailableError(f"{service_id} has no slot at {option.appointment_start}")
    legs = _transport_legs(option) if "transport" in asked else []
    _check_transport(legs, simulated)
    stay = None
    if "accommodation" in asked:
        assert option.accommodation_id and option.outbound_date  # _needs_stay guaranteed these
        stay = (option.accommodation_id, option.outbound_date, option.nights)
        town = simulated.accommodation.place(option.accommodation_id).town
        found = simulated.accommodation.search(town, option.outbound_date, option.nights)
        # The place exists (place() would have refused), so it is in its own town's search.
        chosen = next(o for o in found.options if o.accommodation_id == stay[0])
        if not chosen.available:
            raise NotAvailableError(f"{stay[0]} from {option.outbound_date}: {chosen.reason}")

    # Then make the requests: the appointment, the transport, the stay.
    made = []
    if "appointment" in asked:
        made.append(
            simulated.calendar.request_booking(service_id, option.appointment_start, approval)
        )
    for route_id, day in legs:
        made.append(simulated.transport.request_booking(route_id, day, option.travellers, approval))
    if stay is not None:
        made.append(simulated.accommodation.request_booking(*stay, approval))
    return made
