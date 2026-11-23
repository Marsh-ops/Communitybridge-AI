"""Input and output models for the rules-engine gates.

The gates are the yes/no checks that come before any travel or cost calculation:
residency (§6.1), APY Lands (§6.1.1), recognised and approved specialists (§6.4.1, §6.4.2, §9.2),
ineligible health professionals (§6.4.4) and ineligible appointments (§6.4.5).

Like the distance tool, they take one validated input and return one read-only result, so
no agent can alter a gate's answer afterwards (the design approach: only deterministic tools decide).

Each field below is a *fact the client or coordinator confirms* (Gate 1 in the design diagram), or a
fact looked up in the synthetic dataset. The values are closed lists (`Literal`), not free text,
because a free-text typo such as "on holiday" for "on holidays" would otherwise slip through as
"no problem found". A value outside the list is rejected when the model is built.
"""

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from communitybridge.models.common import NonEmptyStr, StrictModel
from communitybridge.models.eligibility import RuleHit
from communitybridge.models.request import IndigenousStatus

# §6.1: the situations the PDF names. "Itinerant workers ... will be considered" is discretionary,
# so it is routed to an authorised officer rather than decided.
ResidencySituation = Literal["on_holidays", "visiting_family_and_friends", "itinerant_worker"]

# §6.4.5. The PDF excludes "clinical trials or experimental treatments other than phase IV
# trials", so the two kinds of trial need separate values.
AppointmentType = Literal[
    "standard",
    "second_opinion",
    "cosmetic",
    "monitoring",
    "clinical_trial",
    "clinical_trial_phase_iv",
]

GateStatus = Literal["pass", "fail", "needs_review"]


class GateFacts(StrictModel):
    """What the gates need to know about one client and one appointment."""

    model_config = ConfigDict(frozen=True)

    # §6.1
    permanent_sa_resident: bool = True
    residency_situation: ResidencySituation | None = None
    # §6.1.1: whether the client's home town is on the APY Lands comes from the town record.
    lives_on_apy_lands: bool = False
    indigenous_status: IndigenousStatus = "not_stated"
    # §6.4.1, §6.4.2, §9.2: the specialty or service sought, in the PDF's own words.
    specialty: NonEmptyStr
    subspecialty: NonEmptyStr | None = None
    # §6.4.4: the kind of health professional, looked up in the dataset. None means a medical
    # specialist (the normal case).
    professional_category: NonEmptyStr | None = None
    # §6.4.5
    appointment_type: AppointmentType = "standard"


def status_for(rule_hits: list[RuleHit]) -> GateStatus:
    """A failed gate decides the matter; an open review means no final yes yet."""
    outcomes = {hit.outcome for hit in rule_hits}
    if "fail" in outcomes:
        return "fail"
    if "needs_review" in outcomes:
        return "needs_review"
    return "pass"


class GateAssessment(StrictModel):
    """Result of all the gates. Read-only, like the engine's Decision."""

    model_config = ConfigDict(frozen=True)

    status: GateStatus
    rule_hits: list[RuleHit] = Field(min_length=1)

    @model_validator(mode="after")
    def _status_matches_hits(self) -> Self:
        expected = status_for(self.rule_hits)
        if self.status != expected:
            raise ValueError(f"status {self.status!r} does not match the rule hits ({expected!r})")
        return self
