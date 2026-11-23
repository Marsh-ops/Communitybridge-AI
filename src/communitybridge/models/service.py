"""A specialist service record from the curated, community-administered dataset.

Used by the Service Discovery Agent. Each record carries its `source` and `last_verified`
date so the Verifier can reject stale information. `synthetic` is typed as `Literal[True]`,
so a record that is not synthetic cannot even be constructed.
"""

from datetime import date
from typing import Literal

from communitybridge.models.common import NonEmptyStr, StrictModel

# Resident, visiting and telehealth services all count when finding the nearest service
# (PATS §6.4.6), so the mode is part of the record rather than a separate table.
ServiceMode = Literal["resident", "visiting", "telehealth"]


class Service(StrictModel):
    service_id: NonEmptyStr
    name: NonEmptyStr
    specialty: NonEmptyStr
    town: NonEmptyStr
    mode: ServiceMode
    source: NonEmptyStr
    last_verified: date
    synthetic: Literal[True]
