"""Load the three simulated services from their fixtures.

The three services share one `SimulatedActionLog`, so the evaluation can count every simulated
booking request the prototype made and the approval each was made under (the success target
"zero unapproved external actions").
"""

from dataclasses import dataclass
from pathlib import Path

from communitybridge.config import DATA_DIR
from communitybridge.db.seed import SeedDataError, load_accommodation
from communitybridge.models.simulated import AvailabilityRecord, RouteRecord, SlotRecord
from communitybridge.simulated.accommodation_api import SimulatedAccommodationApi
from communitybridge.simulated.calendar_api import SimulatedCalendarApi
from communitybridge.simulated.common import (
    SimulatedActionLog,
    SimulatedDataError,
    read_fixture,
    read_fixture_date,
)
from communitybridge.simulated.transport_api import SimulatedTransportApi


@dataclass(frozen=True)
class SimulatedServices:
    transport: SimulatedTransportApi
    accommodation: SimulatedAccommodationApi
    calendar: SimulatedCalendarApi
    log: SimulatedActionLog


def load_simulated_services(data_dir: Path = DATA_DIR) -> SimulatedServices:
    """Build fresh simulated services (and an empty action log) from the fixtures in `data_dir`."""
    simulated = data_dir / "simulated"
    routes = read_fixture(simulated / "transport.yaml", "routes", RouteRecord)
    availability = read_fixture(
        simulated / "accommodation.yaml", "availability", AvailabilityRecord
    )
    slots = read_fixture(simulated / "calendar.yaml", "slots", SlotRecord)
    try:
        places = load_accommodation(data_dir / "services" / "accommodation.yaml")
    except SeedDataError as exc:
        raise SimulatedDataError(f"accommodation dataset: {exc}") from exc

    log = SimulatedActionLog()
    return SimulatedServices(
        transport=SimulatedTransportApi(
            routes, log, last_verified=read_fixture_date(simulated / "transport.yaml")
        ),
        accommodation=SimulatedAccommodationApi(places, availability, log),
        calendar=SimulatedCalendarApi(slots, log),
        log=log,
    )
