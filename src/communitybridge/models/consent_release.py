"""What the Consent tool shows and returns.

The Consent Agent shows the minimum personal data each action needs, waits for explicit approval
for that action, and lets consent be withdrawn. Nothing here is ever really
sent: every recipient in the policy is a SIMULATED provider or a prospective partner, and the
prototype makes no real data release. So each of these is a
`SimulatedResponse` and carries `simulated: true` and the SIMULATED label.
"""

from datetime import datetime
from typing import Literal

from pydantic import ConfigDict, Field

from communitybridge.models.common import NonEmptyStr, StrictModel
from communitybridge.models.consent import ConsentRequest
from communitybridge.models.consent_policy import RecipientType, Sensitivity
from communitybridge.models.contact import ClientContact
from communitybridge.models.plan import LogisticsOption
from communitybridge.models.request import GoalRequest
from communitybridge.models.service import Service
from communitybridge.models.simulated import BookingRequest, SimulatedResponse


class PromptField(SimulatedResponse):
    """One piece of personal data an action would share, exactly as the user will see it."""

    name: NonEmptyStr
    description: NonEmptyStr
    sensitivity: Sensitivity
    value: str | None = None  # None while the system doesn't have it yet


class ConsentPrompt(SimulatedResponse):
    """The consent screen for one action: what is shared, with whom, why, and what undoes it."""

    action: NonEmptyStr
    description: NonEmptyStr
    recipient: NonEmptyStr
    recipient_type: RecipientType
    purpose: NonEmptyStr
    fields: list[PromptField] = Field(min_length=1)
    missing: list[NonEmptyStr] = Field(default_factory=list)  # fields still to be asked for
    approval_required: Literal[True] = True  # no action can release data without approval
    on_withdrawal: NonEmptyStr  # shown next to the withdraw button


class DataRelease(SimulatedResponse):
    """The data released under an approved consent: the minimum fields and nothing else."""

    model_config = ConfigDict(frozen=True)

    action: NonEmptyStr
    recipient: NonEmptyStr
    recipient_type: RecipientType
    data: dict[str, str] = Field(min_length=1)
    consent_decided_at: datetime  # the approval this release was made under


class WithdrawalNotice(SimulatedResponse):
    """What the user is told after withdrawing consent."""

    action: NonEmptyStr
    recipient: NonEmptyStr
    withdrawn_at: datetime
    message: NonEmptyStr  # what withdrawing undoes for this action, from the policy


class ReleaseRecord(SimulatedResponse):
    """A release as the SIMULATED recipient recorded it: the data, and the approval it was under.

    This is what the SIMULATED action log holds, the data-release twin of a booking request.
    The reference is `SIM-REL-nnnn`; `approval_id` is the Gate 4 approval event in the trace.
    """

    reference: str = Field(pattern=r"^SIM-REL-\d{4}$")
    release: DataRelease
    approval_id: NonEmptyStr


class ConsentSubject(StrictModel):
    """What the Consent tools read to fill in an action's fields: the journey the user chose.

    The Consent Agent hands these four things to its tools and never reads a value from them
    itself, so the personal data stays inside the tools. Read-only.
    """

    model_config = ConfigDict(frozen=True)

    goal: GoalRequest
    service: Service
    option: LogisticsOption
    contact: ClientContact | None = None


class WithdrawalResult(StrictModel):
    """Everything a withdrawal changed: the withdrawn record (the approval stays on it),
    what the user is told, and the SIMULATED holds that were cancelled."""

    model_config = ConfigDict(frozen=True)

    request: ConsentRequest
    notice: WithdrawalNotice
    cancelled: tuple[BookingRequest, ...] = ()
