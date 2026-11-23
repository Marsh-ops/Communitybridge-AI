"""Journey planner: the logistics calculator and the window matcher, as options.

This is the one read-only tool the Logistics Agent uses to plan. It puts together two tools that
already exist and are tested on their own:

1. `assess_logistics`: which way of travelling PATS pays for, the stay, the
   entitlements, the questions to ask. Rates and limits come from the rules data, fares from the
   SIMULATED fare sheet.
2. `plan_windows`: for each appointment slot in the client's dates, the transport
   and stay that work with it, and for the slots that don't, every reason why.

and turns each window that works into a `LogisticsOption`, with the citations a plan statement will
need: the PATS sections behind the costs, the service record, the simulated route and
the accommodation record, each with the date it was last checked. It never decides eligibility
(the core rule: it refuses a client the engine did not find eligible) and it books nothing: a booking
is a separate tool that needs an approval (`tools/booking.py`).
"""

from collections.abc import Mapping

from communitybridge.models.common import Citation
from communitybridge.models.dataset import Accommodation
from communitybridge.models.engine import EligibilityResult
from communitybridge.models.facts import ConfirmedFacts
from communitybridge.models.journey import JourneyPlan
from communitybridge.models.logistics import LogisticsAssessment
from communitybridge.models.plan import LogisticsOption
from communitybridge.models.request import GoalRequest
from communitybridge.models.windows import WindowOption
from communitybridge.simulated.services import SimulatedServices
from communitybridge.tools.logistics import LogisticsError, assess_logistics
from communitybridge.tools.rule_access import norm
from communitybridge.tools.rules_engine import EngineContext
from communitybridge.tools.verifier import citation_for_section
from communitybridge.tools.windows import plan_windows

TRANSPORT_SOURCE = "SIMULATED transport timetable (data/simulated/transport.yaml)"


def _section_key(section: str) -> tuple[int, ...]:
    return tuple(int(part) for part in section.split("."))


def _citations(
    window: WindowOption,
    assessment: LogisticsAssessment,
    service_id: str,
    simulated: SimulatedServices,
    accommodation: Mapping[str, Accommodation],
    context: EngineContext,
) -> list[Citation]:
    """Every source behind one option, each with the date it was last checked."""
    hits = list(assessment.modes.rule_hits)
    if window.accommodation is not None:
        hits += window.accommodation.rule_hits
    sections = sorted({hit.section for hit in hits}, key=_section_key)
    citations = [citation_for_section(section, context.rules) for section in sections]

    service = next(s for s in context.services if s.service_id == service_id)
    citations.append(
        Citation(
            source=service.source,
            section_or_record=f"service {service.service_id}",
            last_verified=service.last_verified,
        )
    )
    route_ids = [r for r in (window.outbound_route_id, window.return_route_id) if r is not None]
    for route_id in dict.fromkeys(route_ids):
        checked = simulated.transport.last_verified
        if checked is None:
            raise LogisticsError("the simulated transport fixture has no last_verified date")
        citations.append(
            Citation(
                source=TRANSPORT_SOURCE,
                section_or_record=f"route {simulated.transport.route(route_id).route_id}",
                last_verified=checked,
            )
        )
    if window.accommodation_id is not None:
        record = accommodation.get(window.accommodation_id)
        if record is None:
            raise LogisticsError(f"accommodation {window.accommodation_id} has no dataset record")
        citations.append(
            Citation(
                source=record.source,
                section_or_record=f"accommodation {record.accommodation_id}",
                last_verified=record.last_verified,
            )
        )
    return citations


def plan_journey(
    result: EligibilityResult,
    goal: GoalRequest,
    facts: ConfirmedFacts,
    destination_service_id: str,
    simulated: SimulatedServices,
    context: EngineContext,
    accommodation: Mapping[str, Accommodation],
    include_fallback: bool = False,
) -> JourneyPlan:
    """Plan the travel for an eligible client and list the windows that work, as options.

    Raises `LogisticsError` if the client is not eligible or a town or service is unknown, and
    the named errors of the tools underneath (`DistanceNotFoundError`, `RuleParameterError`). The
    telehealth fallback is left out unless `include_fallback` is True, because only the
    Coordinator's re-planning may ask for it.
    """
    destination = next(
        (s for s in context.services if s.service_id == destination_service_id), None
    )
    origin = context.towns.get(norm(goal.origin_town))
    if destination is None or origin is None:
        raise LogisticsError("the destination service or the origin town is not known")

    fares = simulated.transport.fares(origin.town_id, destination.town)
    assessment = assess_logistics(result, goal, facts, destination_service_id, fares, context)
    matches = plan_windows(
        result,
        assessment,
        goal,
        facts,
        destination_service_id,
        simulated,
        context,
        include_fallback,
    )
    options = [
        LogisticsOption(
            travel_mode=window.travel_mode or "telehealth",
            travellers=assessment.travellers,
            accommodation_id=window.accommodation_id,
            appointment_start=window.appointment.starts,
            citations=_citations(
                window, assessment, destination_service_id, simulated, accommodation, context
            ),
            simulated=True,
            outbound_date=window.outbound_date,
            return_date=window.return_date,
            nights=window.nights,
            outbound_route_id=window.outbound_route_id,
            return_route_id=window.return_route_id,
            accommodation_name=window.accommodation_name,
            accommodation_total_aud=window.accommodation_total_aud,
            accommodation_subsidy_aud=(
                window.accommodation.subsidy_aud if window.accommodation else None
            ),
            travel_subsidy_aud=window.travel_cost_aud,
            notes=window.notes,
        )
        for window in matches.options
    ]
    return JourneyPlan(assessment=assessment, options=options, rejected=matches.rejected)
