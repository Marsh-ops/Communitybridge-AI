"""Eligibility models: the rules engine's Decision and the Eligibility Agent's explanation.

This is where the most important design rule is enforced in the data itself: only the rules engine decides eligibility.

- `RuleHit` and `Decision` are frozen, so no agent can change a decision after the rules engine
  has produced it.
- `EligibilityExplanation`, the Eligibility Agent's output, has no `eligible` field, and
  `StrictModel` forbids adding one. The LLM can only explain a decision; it cannot alter it.
"""

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from communitybridge.models.common import SECTION_PATTERN, NonEmptyStr, StrictModel

# "needs_review" is for discretionary PDF wording (e.g. §6.1 "will be considered"): the engine
# never guesses, and the case goes to a PATS authorised officer (§8.3).
RuleOutcome = Literal["pass", "fail", "not_applicable", "needs_review"]


class RuleHit(StrictModel):
    """One PATS rule the engine evaluated, with the guideline section it comes from."""

    model_config = ConfigDict(frozen=True)

    # Format from the project design, e.g. "PATS-6.4.6-NEAREST".
    rule_id: str = Field(pattern=r"^PATS-\d+(\.\d+)*-[A-Z0-9]+(-[A-Z0-9]+)*$")
    section: str = Field(pattern=SECTION_PATTERN)
    outcome: RuleOutcome
    reason: NonEmptyStr

    @model_validator(mode="after")
    def _rule_id_matches_section(self) -> Self:
        # Catches a rule cited under the wrong section, which would mislead the user.
        if not self.rule_id.startswith(f"PATS-{self.section}-"):
            raise ValueError(f"rule_id {self.rule_id} does not belong to section {self.section}")
        return self


class Decision(StrictModel):
    """The rules engine's output. Read-only input for every agent after it."""

    model_config = ConfigDict(frozen=True)

    eligible: bool
    rule_hits: list[RuleHit] = Field(min_length=1)
    rules_version: NonEmptyStr

    @model_validator(mode="after")
    def _outcome_matches_rule_hits(self) -> Self:
        outcomes = {hit.outcome for hit in self.rule_hits}
        # A "not eligible" result must point to the rule behind it, so the explanation can
        # always cite the reason (no uncited statements, the core rule).
        if not self.eligible and not outcomes & {"fail", "needs_review"}:
            raise ValueError(
                "an ineligible decision needs a rule hit with outcome fail or needs_review"
            )
        # An open review means no final yes yet.
        if self.eligible and "needs_review" in outcomes:
            raise ValueError("a decision with a needs_review hit cannot be eligible")
        return self


class EligibilityExplanation(StrictModel):
    """Plain-language wording of the decision. No `eligible` field, by design.

    Two things in the graph state have this shape: `reasons`, built by the rules from the
    engine's own rule reasons and shown as the plan's verified statement, and `explanation`, the
    wording a language model wrote, which is shown apart from the plan under `MODEL_WORDING_LABEL`
    and is never a plan statement.
    """

    text: NonEmptyStr
    cited_sections: list[str] = Field(min_length=1)


# What the user is told about a language model's wording of the decision. It says
# exactly what was checked, because the check is real but narrow: the sections it cites are in the
# decision and the outcome it states is the engine's. It cannot judge a claim about the person.
MODEL_WORDING_LABEL = (
    "Written by a language model. It was checked only for the sections it cites and the outcome "
    "it states, not for what it says about you, so it is not verified."
)
