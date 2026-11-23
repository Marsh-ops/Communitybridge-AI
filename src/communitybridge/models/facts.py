"""The details a client or coordinator confirms beyond the goal text (Gate 1 in the design diagram).

The Coordinator Agent extracts origin, specialty, dates and escort needs from the plain-language
goal (`GoalRequest`). Everything else the rules need, such as whether the client holds a
concession card, their age, or why they want to travel past a nearer service, is not in the goal.
The client or coordinator confirms it, and it lands here.

The rules engine takes a `GoalRequest` plus these `ConfirmedFacts`. The 30 evaluation scenarios
use the same class, so the engine is tested with exactly the input it gets in the app.

`extra="forbid"` (from `StrictModel`) means a misspelt fact name fails validation instead of being
silently ignored, which would otherwise make the engine decide on the wrong facts. The defaults
describe a typical adult client with no special circumstances.
"""

from pydantic import Field

from communitybridge.models.common import NonEmptyStr, StrictModel
from communitybridge.models.entitlements import EscortCriterion
from communitybridge.models.gates import AppointmentType, ResidencySituation
from communitybridge.models.logistics import AirCriterion
from communitybridge.models.nearest import BypassReason
from communitybridge.models.request import IndigenousStatus


class ConfirmedFacts(StrictModel):
    """Confirmed details beyond the goal text. Defaults describe a typical adult client."""

    # None = use the identity extracted from the goal text.
    indigenous_status: IndigenousStatus | None = None

    # §6.1 and §6.5: residency, and travel that starts away from home.
    permanent_sa_resident: bool = True
    residency_situation: ResidencySituation | None = None
    away_from_residence: bool = False

    client_age: int = Field(default=45, ge=0)

    # §6.5.5 advance and §7.3.1 accommodation.
    concession_card: bool = False
    severe_financial_hardship: bool = False
    notice_business_days: int | None = Field(default=None, ge=0)

    # §6.4.5, and the treatment performed (§6.5.3); None = the specialty sought.
    appointment_type: AppointmentType = "standard"
    treatment: str | None = None
    subspecialty: NonEmptyStr | None = None
    alternate_route_km: float | None = Field(default=None, ge=0)

    # §6.4.6 and §8.1.1: why the client travels past a nearer service.
    bypass_reason: BypassReason | None = None
    gp_reason_documented: bool = False
    local_timeframe_clinically_acceptable: bool = True
    visiting_primary_rooms_service_id: str | None = None
    # "cannot be treated in South Australia" referrals are only accepted from a SA Health
    # tertiary health site. None = not stated.
    referred_from_tertiary_site: bool | None = None

    # §7.2.8 and §7.2.9: escort approval.
    escort_criterion: EscortCriterion | None = None
    escort_specialist_approved: bool = False

    # §7.2.5, §7.2.6 and §6.5.8: used by the logistics calculator.
    air_medical_criterion: AirCriterion | None = None
    air_specialist_authorised: bool = False
    requested_modes: list[str] = Field(default_factory=list)
    # A private vehicle is an eligible surface mode (§7.2.1). False only where a scenario needs
    # the client to have no car, e.g. before the telehealth fallback is considered.
    has_private_vehicle: bool = True

    # §6.6.1 and §7.3.1: accommodation. Staying with family or friends is never subsidised, and
    # more than the automatic two nights needs the treating specialist's authorisation.
    staying_with_family_or_friends: bool = False
    specialist_authorised_nights: int | None = Field(default=None, ge=1)
