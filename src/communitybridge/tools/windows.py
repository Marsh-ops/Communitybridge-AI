"""Appointment-window matching: slots, transport and stays that work together.

Serves the Logistics Agent. The project plan's fourth in-scope step is to suggest "transport,
accommodation and appointment windows that work together". For each appointment slot in the
client's dates this tool checks the SIMULATED transport and accommodation, and reports the
arrangements that work and, for the slots that don't, every reason why. The Coordinator's
re-planning follows those reasons: the next window, then the telehealth fallback.

It only reads the SIMULATED services. It books nothing; a booking is a separate step that needs an
approval (no real bookings, and the approval gates in the graph).

Planning conventions. These are not PATS rules; they are kept in this one place:
- The client travels out the day before the appointment (two nights), or on the day (one night),
  and comes home the day after. A stay never exceeds the nights claimable without medical
  authorisation (§7.3.1, read from the rules data).
- Travelling on the day only works if the transport arrives before the appointment starts.
  Driving has no timetable in the data, so travel time by road is not checked (and says so).
- Accommodation must be commercial (§6.6.1). The cheapest one with a room for every night is used.
- Telehealth needs no travel or stay. A telehealth slot appears only if the calendar search the
  caller passed in included it (the fallback is hidden unless asked for).
"""

from datetime import date, time, timedelta

from communitybridge.models.engine import EligibilityResult
from communitybridge.models.facts import ConfirmedFacts
from communitybridge.models.logistics import (
    AccommodationSubsidyInput,
    LogisticsAssessment,
)
from communitybridge.models.request import GoalRequest
from communitybridge.models.rules import RuleSet
from communitybridge.models.simulated import (
    AccommodationOption,
    AppointmentSlot,
    CalendarSearch,
    TransportOption,
)
from communitybridge.models.windows import (
    RejectedWindow,
    WindowMatches,
    WindowOption,
    WindowRequest,
)
from communitybridge.simulated.accommodation_api import SimulatedAccommodationApi
from communitybridge.simulated.services import SimulatedServices
from communitybridge.simulated.transport_api import SimulatedTransportApi
from communitybridge.tools.logistics import (
    NIGHTS_RULE,
    LogisticsError,
    accommodation_subsidy,
    public_transport_modes,
)
from communitybridge.tools.rule_access import find_rule, norm, number
from communitybridge.tools.rules_engine import EngineContext


def _find_route(
    transport: SimulatedTransportApi,
    from_town: str,
    to_town: str,
    travel_date: date,
    wanted_modes: set[str],
    label: str,
    arrive_before: time | None,
) -> tuple[TransportOption | None, str | None]:
    """The cheapest route of the chosen kind that runs that day (and arrives in time)."""
    options = [
        o
        for o in transport.search(from_town, to_town, travel_date).options
        if o.mode in wanted_modes
    ]
    if not options:
        return None, f"no {label} route between {from_town} and {to_town} (SIMULATED timetable)"
    running = [o for o in options if o.available]
    if not running:
        why = "; ".join(f"{o.route_id}: {o.reason}" for o in options)
        return None, f"no {label} on {travel_date}: {why}"
    if arrive_before is not None:
        in_time = [o for o in running if time.fromisoformat(o.arrive) < arrive_before]
        if not in_time:
            late = ", ".join(f"{o.route_id} arrives {o.arrive}" for o in running)
            return None, (
                f"the {label} on {travel_date} gets there too late for the "
                f"{arrive_before:%H:%M} appointment ({late})"
            )
        running = in_time
    return min(running, key=lambda o: (o.fare_aud, o.route_id)), None


def _find_stay(
    accommodation: SimulatedAccommodationApi, town: str, check_in: date, nights: int
) -> tuple[AccommodationOption | None, str | None]:
    """The cheapest commercial place (§6.6.1) with a room for every night."""
    options = [o for o in accommodation.search(town, check_in, nights).options if o.commercial]
    if not options:
        return None, f"no commercial accommodation in {town} (SIMULATED)"
    rooms = [o for o in options if o.available]
    if not rooms:
        why = "; ".join(f"{o.accommodation_id}: {o.reason}" for o in options)
        return None, (
            f"no commercial accommodation has a room for {nights} night(s) from {check_in}: {why}"
        )
    return min(rooms, key=lambda o: (o.total_aud, o.accommodation_id)), None


def _try_arrangement(
    slot: AppointmentSlot,
    outbound: date,
    nights: int,
    request: WindowRequest,
    transport: SimulatedTransportApi,
    accommodation: SimulatedAccommodationApi,
    wanted_modes: set[str],
    label: str,
    rules: RuleSet,
) -> tuple[WindowOption | None, list[str]]:
    """One way of fitting the slot: out on `outbound`, `nights` nights, home the day after."""
    appointment_day = slot.starts.date()
    home = appointment_day + timedelta(days=1)
    problems: list[str] = []
    notes: list[str] = []
    out_route = back_route = None

    if request.travel_mode == "private_vehicle":
        notes.append("Travel time by road is not checked (SIMULATED planning).")
    else:
        # On the day itself the transport must arrive before the appointment starts.
        in_time = slot.starts.time() if outbound == appointment_day else None
        out_route, problem = _find_route(
            transport, request.origin_town_id, request.destination_town_id,
            outbound, wanted_modes, label, in_time,
        )  # fmt: skip
        if problem:
            problems.append(problem)
        back_route, problem = _find_route(
            transport, request.destination_town_id, request.origin_town_id,
            home, wanted_modes, label, None,
        )  # fmt: skip
        if problem:
            problems.append(problem)

    stay = subsidy = None
    if request.staying_with_family_or_friends:
        notes.append("Staying with family or friends: there is no accommodation subsidy (§6.6.1).")
    else:
        stay, problem = _find_stay(accommodation, request.destination_town_id, outbound, nights)
        if problem:
            problems.append(problem)
        elif stay is not None:
            subsidy = accommodation_subsidy(
                AccommodationSubsidyInput(
                    nights=nights,
                    travellers=request.travellers,
                    concession_card=request.concession_card,
                    first_night_waiver=request.first_night_waiver,
                    nightly_cost_aud=stay.nightly_rate_aud,
                ),
                rules,
            )

    if problems:
        return None, [f"Travelling on {outbound} ({nights} night(s)): {p}" for p in problems]
    return (
        WindowOption(
            appointment=slot,
            travel_mode=request.travel_mode,
            outbound_date=outbound,
            return_date=home,
            nights=nights,
            outbound_route_id=out_route.route_id if out_route else None,
            return_route_id=back_route.route_id if back_route else None,
            accommodation_id=stay.accommodation_id if stay else None,
            accommodation_name=stay.name if stay else None,
            accommodation_total_aud=stay.total_aud if stay else None,
            accommodation=subsidy,
            travel_cost_aud=request.travel_cost_aud,
            notes=notes,
        ),
        [],
    )


def _telehealth_option(slot: AppointmentSlot) -> WindowOption:
    notes = ["No travel or accommodation is needed for a telehealth appointment (SIMULATED slot)."]
    if slot.offered_as == "fallback_only":
        notes.append(
            "Offered only because travel couldn't be arranged: telehealth is a last line of "
            "defence."
        )
    return WindowOption(
        appointment=slot,
        travel_mode=None,
        outbound_date=None,
        return_date=None,
        nights=0,
        notes=notes,
    )


def match_windows(
    request: WindowRequest,
    slots: CalendarSearch,
    transport: SimulatedTransportApi,
    accommodation: SimulatedAccommodationApi,
    rules: RuleSet,
) -> WindowMatches:
    """Sort every slot in the search into the arrangements that work and the reasons why not."""
    automatic = int(number(find_rule(rules, NIGHTS_RULE), "nights_without_medical_authorisation"))
    wanted_modes = {"air"} if request.travel_mode == "air" else public_transport_modes(rules)
    label = "flight" if request.travel_mode == "air" else "public transport"

    options: list[WindowOption] = []
    rejected: list[RejectedWindow] = []
    for slot in slots.slots:
        if slot.mode == "telehealth":
            options.append(_telehealth_option(slot))
            continue
        if request.travel_mode is None:
            rejected.append(
                RejectedWindow(
                    appointment=slot,
                    reasons=["The travel mode isn't decided yet, so this slot can't be matched."],
                )
            )
            continue

        day = slot.starts.date()
        # Out the day before (two nights) or on the day (one night); never more than the
        # nights claimable without medical authorisation (§7.3.1).
        arrangements = [(day - timedelta(days=1), 2), (day, 1)]
        reasons: list[str] = []
        found = []
        for outbound, nights in arrangements:
            if nights > automatic:
                continue
            option, problems = _try_arrangement(
                slot,
                outbound,
                nights,
                request,
                transport,
                accommodation,
                wanted_modes,
                label,
                rules,
            )
            if option is not None:
                found.append(option)
            reasons.extend(problems)
        if found:
            options.extend(found)
        else:
            rejected.append(
                RejectedWindow(
                    appointment=slot,
                    reasons=reasons
                    or ["No arrangement fits the nights allowed without authorisation (§7.3.1)."],
                )
            )

    options.sort(key=lambda o: (o.appointment.starts, o.outbound_date or date.min))
    return WindowMatches(options=options, rejected=rejected)


def plan_windows(
    result: EligibilityResult,
    plan: LogisticsAssessment,
    goal: GoalRequest,
    facts: ConfirmedFacts,
    destination_service_id: str,
    simulated: SimulatedServices,
    context: EngineContext,
    include_fallback: bool = False,
) -> WindowMatches:
    """Match the destination's appointment slots in the client's dates, for an eligible client.

    The telehealth fallback is left out unless `include_fallback` is True, because only the
    Coordinator's re-planning may ask for it.
    """
    if result.outcome != "eligible" or result.entitlements is None:
        raise LogisticsError("windows are only matched for a client the engine found eligible")
    destination = next(
        (s for s in context.services if s.service_id == destination_service_id), None
    )
    origin = context.towns.get(norm(goal.origin_town))
    if destination is None or origin is None:
        raise LogisticsError("the destination service or the origin town is not known")

    request = WindowRequest(
        origin_town_id=origin.town_id,
        destination_town_id=destination.town,
        travellers=plan.travellers,
        travel_mode=plan.modes.chosen_mode,
        travel_cost_aud=plan.modes.travel_subsidy_aud,
        concession_card=facts.concession_card,
        first_night_waiver=result.entitlements.first_night_concession_waived,
        staying_with_family_or_friends=facts.staying_with_family_or_friends,
    )
    slots = simulated.calendar.search(
        destination_service_id, goal.earliest_date, goal.latest_date, include_fallback
    )
    return match_windows(
        request, slots, simulated.transport, simulated.accommodation, context.rules
    )
