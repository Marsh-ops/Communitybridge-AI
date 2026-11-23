"""The structured request the Coordinator Agent extracts from the user's plain-language goal.

The project plan's first in-scope step is to extract origin, specialty, dates and escort needs
into a validated request that the user confirms. `confirmed_by_user` therefore defaults to
False: an LLM extraction is never treated as confirmed until the user says so at Gate 1.

There are two models, one inside the other. `ExtractedGoal` is the six details the language
model is asked to fill in. `GoalRequest` is the same six plus `confirmed_by_user`. The model is
only ever given `ExtractedGoal`, so it has no field in which to confirm on the user's behalf, and
because every boundary model forbids unknown fields, a reply that adds `confirmed_by_user` is
refused instead of being accepted and trusted.
"""

from datetime import date
from typing import Literal, Self

from pydantic import model_validator

from communitybridge.models.common import NonEmptyStr, StrictModel

# Aboriginal and Torres Strait Islander identity are kept separate because the PATS PDF
# treats them differently: the APY Lands exclusion (§6.1.1) names "Aboriginal people" only,
# while the Aboriginal Engagement Strategy (§6.1.2) and escort table (§7.2.9) cover both.
# "not_stated" is not the same as "neither": the engine must ask, not assume.
IndigenousStatus = Literal["aboriginal", "torres_strait_islander", "both", "neither", "not_stated"]


class ExtractedGoal(StrictModel):
    """What the language model fills in: six details, and no way to confirm them."""

    origin_town: NonEmptyStr
    specialty: NonEmptyStr
    earliest_date: date
    latest_date: date
    escort_requested: bool = False
    # Sensitive data, so the Consent Agent treats it as minimum-necessary.
    indigenous_status: IndigenousStatus = "not_stated"

    @model_validator(mode="after")
    def _dates_in_order(self) -> Self:
        if self.latest_date < self.earliest_date:
            raise ValueError("latest_date must be on or after earliest_date")
        return self


class GoalRequest(ExtractedGoal):
    """The six details plus whether the user has confirmed them at Gate 1."""

    confirmed_by_user: bool = False
