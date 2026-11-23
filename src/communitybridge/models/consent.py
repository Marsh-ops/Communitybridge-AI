"""A per-action consent request, as prepared by the Consent Agent.

The project plan says the system shows the minimum personal data each action needs, waits for
approval and allows consent to be withdrawn. `approved` has three states: None (pending), True or
False. A decision must be timestamped and a pending request must not be, so the trace shows
exactly when the user decided. The "zero unapproved external actions" success target is measured
from these records.

Withdrawal is a fourth state on top of an approval, recorded by `withdrawn_at`. Only an
approved consent can be withdrawn, because there is nothing to undo otherwise. `withdraw()`
returns a new record instead of changing this one, so the approval and the withdrawal both stay
visible in the trace.
"""

from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from communitybridge.models.common import NonEmptyStr, StrictModel

ConsentStatus = Literal["pending", "declined", "approved", "withdrawn"]


class ConsentRequest(StrictModel):
    action: NonEmptyStr
    # Field names come from data/policy/data_fields.yaml, never from the LLM.
    fields_required: list[NonEmptyStr] = Field(min_length=1)
    purpose: NonEmptyStr
    approved: bool | None = None
    decided_at: datetime | None = None
    withdrawn_at: datetime | None = None

    @model_validator(mode="after")
    def _decision_and_timestamps_are_consistent(self) -> Self:
        if (self.approved is None) != (self.decided_at is None):
            raise ValueError("decided_at must be set exactly when approved is True or False")
        if self.withdrawn_at is not None:
            if self.approved is not True:
                raise ValueError("only an approved consent can be withdrawn")
            # decided_at is set here: the check above guarantees it when approved is True.
            if self.decided_at is not None and self.withdrawn_at < self.decided_at:
                raise ValueError("withdrawn_at cannot be before decided_at")
        return self

    @property
    def status(self) -> ConsentStatus:
        if self.withdrawn_at is not None:
            return "withdrawn"
        if self.approved is None:
            return "pending"
        return "approved" if self.approved else "declined"

    @property
    def is_active(self) -> bool:
        """True only while data may be released under this consent."""
        return self.status == "approved"

    def withdraw(self, at: datetime) -> "ConsentRequest":
        """Return a withdrawn copy. Validated again, so a bad timestamp is still rejected."""
        if self.status == "withdrawn":
            raise ValueError(f"consent for {self.action} is already withdrawn")
        return ConsentRequest.model_validate({**self.model_dump(), "withdrawn_at": at})


# The consent action that covers each kind of SIMULATED booking request, as named in
# data/policy/data_fields.yaml. One place, so the Consent Agent, the booking step, the trace
# audit and the SIMULATED action log cannot disagree about which approval covers which request.
ACTION_FOR_KIND: dict[str, str] = {
    "appointment": "simulated_appointment_request",
    "transport": "simulated_transport_request",
    "accommodation": "simulated_accommodation_request",
}
KIND_FOR_ACTION: dict[str, str] = {action: kind for kind, action in ACTION_FOR_KIND.items()}
