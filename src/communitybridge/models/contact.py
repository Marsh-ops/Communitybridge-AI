"""The client's own contact details, which only the Consent step uses.

Booking an appointment or a seat needs a name, a way to reach the client and (for the clinic) a
date of birth. None of that is in the goal text or the confirmed facts, and the rules engine must
never see it. It is *input*, given when the run starts like `ConfirmedFacts`, and it is read only
by the Consent step, which shows it to the user at Gate 4 before any of it leaves the system.

`synthetic` is required and must be true (the core rule: synthetic data only, no real people), so
a record that is not marked synthetic cannot be built. `mobility_assistance_needed` has no
default on purpose: an assumed "no" would be shown to the user as if they had said it.
"""

from datetime import date
from typing import Literal

from communitybridge.models.common import NonEmptyStr, StrictModel


class ClientContact(StrictModel):
    synthetic: Literal[True]
    given_name: NonEmptyStr
    family_name: NonEmptyStr
    contact_phone: NonEmptyStr
    date_of_birth: date
    mobility_assistance_needed: bool
