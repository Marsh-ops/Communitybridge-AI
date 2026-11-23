"""Models for the Verifier: the claims it checks and the report it returns.

The core rule: no uncited statements in a plan. The Verifier blocks them, and the UI shows them as
blocked. The Verifier is deterministic (no language model), so everything it decides is a rule
applied to data and can be tested.

A `Claim` is a statement an agent wants in the plan, with the citations it relies on. `subject` and
`value` are optional: two claims about the same subject (for example "the town of service X")
that state different values conflict. `core` marks the claims the whole plan rests on (the
eligibility result, the chosen service, the travel mode); if one of those is blocked the plan goes
back to be re-planned (the design diagram) instead of being shown with a hole in it.
"""

from datetime import date
from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from communitybridge.models.common import Citation, NonEmptyStr, StrictModel
from communitybridge.models.plan import PlanItem

Check = Literal["citation", "freshness", "conflict"]


class Claim(StrictModel):
    """A statement an agent wants in the plan, with the sources it relies on."""

    model_config = ConfigDict(frozen=True)

    statement: NonEmptyStr
    citations: list[Citation] = Field(default_factory=list)
    # What the statement is about and what it says, so two claims can be compared.
    subject: NonEmptyStr | None = None
    value: NonEmptyStr | None = None
    core: bool = False

    @model_validator(mode="after")
    def _subject_and_value_go_together(self) -> Self:
        if (self.subject is None) != (self.value is None):
            raise ValueError("a claim needs both a subject and a value to be compared, or neither")
        return self


class Finding(StrictModel):
    """One reason a claim failed a check, in words the user can read."""

    model_config = ConfigDict(frozen=True)

    check: Check
    detail: NonEmptyStr


class ClaimResult(StrictModel):
    """What the Verifier decided about one claim."""

    model_config = ConfigDict(frozen=True)

    item: PlanItem  # "verified", or "blocked" with the reason the user will see
    findings: list[Finding] = Field(default_factory=list)
    core: bool = False

    @model_validator(mode="after")
    def _blocked_exactly_when_there_are_findings(self) -> Self:
        if (self.item.status == "blocked") != bool(self.findings):
            raise ValueError("a claim is blocked exactly when it has findings")
        return self


class VerificationReport(StrictModel):
    """The Verifier's verdict on every claim. Read-only."""

    model_config = ConfigDict(frozen=True)

    results: list[ClaimResult]
    # Stale or conflicting information the user should be asked to confirm.
    escalations: list[NonEmptyStr] = Field(default_factory=list)
    checked_on: date
    max_age_days: int = Field(ge=1)

    @property
    def items(self) -> list[PlanItem]:
        return [result.item for result in self.results]

    @property
    def blocked(self) -> list[ClaimResult]:
        return [result for result in self.results if result.item.status == "blocked"]

    @property
    def all_verified(self) -> bool:
        return not self.blocked

    @property
    def replan_needed(self) -> bool:
        """A blocked core claim sends the plan back to the Coordinator (the design diagram)."""
        return any(result.core for result in self.blocked)


class PlanCheck(StrictModel):
    """What the Verifier tool returns for a whole plan: the verdict and what to ask the user.

    `questions` joins the two kinds of thing the user is asked at the plan gate, once each: what
    the verdict asks them to confirm (stale or conflicting information) and what the other tools
    asked (an entitlement that depends on an answer, a travel mode PATS does not name). They are
    not statements, so they are not checked for sources.
    """

    model_config = ConfigDict(frozen=True)

    report: VerificationReport
    questions: list[NonEmptyStr] = Field(default_factory=list)
