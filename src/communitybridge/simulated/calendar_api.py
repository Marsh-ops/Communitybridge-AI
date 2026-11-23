"""SIMULATED appointment calendar: the slots each service offers.

Serves the Logistics Agent and the Coordinator's re-planning. The slots are invented in
`data/simulated/calendar.yaml`. Every response is a `SimulatedResponse`, so it carries
`simulated: true` and the label SIMULATED.

Telehealth is a last line of defence (accepted). A listed telehealth service's slots are
returned like any others, but the referred specialist's own telehealth slot (`fallback_only`) is
hidden unless the caller passes `include_fallback=True`. So the Logistics Agent cannot offer it by
accident; only the Coordinator's re-planning, after it has checked that no in-person option is
left, asks for it.
"""

from collections.abc import Sequence
from datetime import date, datetime

from communitybridge.models.simulated import (
    AppointmentSlot,
    BookingRequest,
    CalendarSearch,
    SlotRecord,
)
from communitybridge.simulated.common import (
    InvalidRequestError,
    NotAvailableError,
    SimulatedActionLog,
    require_approval,
)


class SimulatedCalendarApi:
    def __init__(self, slots: Sequence[SlotRecord], log: SimulatedActionLog) -> None:
        self._rows = list(slots)
        self._log = log

    def _slots(self, service_id: str) -> list[AppointmentSlot]:
        return [
            AppointmentSlot(
                service_id=row.service_id,
                starts=start,
                mode=row.mode,
                telehealth_host_town=row.telehealth_host_town,
                offered_as=row.offered_as,
            )
            for row in self._rows
            if row.service_id == service_id
            for start in row.starts
        ]

    def search(
        self,
        service_id: str,
        earliest: date,
        latest: date,
        include_fallback: bool = False,
    ) -> CalendarSearch:
        """The service's slots on or between the two dates, earliest first."""
        if latest < earliest:
            raise InvalidRequestError("the window ends before it starts")
        slots = [
            slot
            for slot in self._slots(service_id)
            if earliest <= slot.starts.date() <= latest
            and (include_fallback or slot.offered_as != "fallback_only")
        ]
        return CalendarSearch(
            service_id=service_id,
            earliest=earliest,
            latest=latest,
            slots=sorted(slots, key=lambda slot: slot.starts),
        )

    def request_booking(
        self, service_id: str, starts: datetime, approval_id: str
    ) -> BookingRequest:
        """Make a SIMULATED request for an exact slot. Needs an approval id; there is no price."""
        approval = require_approval(approval_id)
        slot = next((s for s in self._slots(service_id) if s.starts == starts), None)
        if slot is None:
            raise NotAvailableError(f"{service_id} has no slot at {starts}")
        return self._log.record(
            "appointment",
            f"{slot.mode.replace('_', ' ')} appointment with {service_id} at "
            f"{starts:%Y-%m-%d %H:%M}",
            None,
            approval,
        )
