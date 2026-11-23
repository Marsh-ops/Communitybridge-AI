"""Schema for the versioned PATS rules file (data/rules/pats_v26_4.yaml).

The project plan stores the PATS rules as versioned data, so a guideline update changes the
YAML and not the code (the guidelines changed five times between January 2025 and June 2026,
per §9.6). This schema makes sure every rule can be traced back to the PDF: a section ID as
numbered in the PDF, the PDF heading, a page number and a verbatim quote. The rules engine
 reads parameters from here and never hard-codes a threshold.
"""

from datetime import date
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from communitybridge.models.common import SECTION_PATTERN, NonEmptyStr, StrictModel

RuleType = Literal["gate", "entitlement", "exclusion", "calculation"]


class Rule(StrictModel):
    """One PATS rule as data. `section` is numbered as in the PDF, and `source_quote` is the
    verbatim text it is based on, so every rule can be checked against the guideline."""

    rule_id: str = Field(pattern=r"^PATS-\d+(\.\d+)*-[A-Z0-9]+(-[A-Z0-9]+)*$")
    section: str = Field(pattern=SECTION_PATTERN)
    title: NonEmptyStr  # heading copied from the PDF
    type: RuleType
    parameters: dict[str, Any] = Field(default_factory=dict)
    source_page: int = Field(ge=1)
    source_quote: NonEmptyStr  # verbatim PDF text, all on source_page
    plain_summary: NonEmptyStr
    # Set when the PDF wording needed interpreting; should explain how it was read.
    interpretation: str | None = None

    @model_validator(mode="after")
    def _rule_id_matches_section(self) -> Self:
        if not self.rule_id.startswith(f"PATS-{self.section}-"):
            raise ValueError(f"rule_id {self.rule_id} does not belong to section {self.section}")
        return self


class SectionNote(StrictModel):
    """A cited section that is not a rule: a heading with no body, a reference or a precondition."""

    section: str = Field(pattern=SECTION_PATTERN)
    title: NonEmptyStr
    source_page: int = Field(ge=1)
    source_quote: NonEmptyStr
    note: NonEmptyStr


class OutOfScopeSection(SectionNote):
    """A PATS section the prototype deliberately does not assess (the project plan's scope).

    The engine never evaluates it, but it isn't silently ignored: the plan shows `user_message`
    whenever `shown_when` applies, so a client who might qualify under it is pointed to a person
    instead of being told "not eligible".
    """

    shown_when: Literal["distance_requirement_not_met", "nearest_service_requirement_not_met"]
    user_message: NonEmptyStr


class RuleSet(StrictModel):
    guideline_version: NonEmptyStr
    source_file: NonEmptyStr
    effective_date: date | None
    effective_date_note: str | None = None
    last_verified: date
    decision_records: list[NonEmptyStr] = Field(default_factory=list)
    heading_only_sections: list[SectionNote] = Field(default_factory=list)
    referenced_sections: list[SectionNote] = Field(default_factory=list)
    assumed_preconditions: list[SectionNote] = Field(default_factory=list)
    out_of_scope_sections: list[OutOfScopeSection] = Field(default_factory=list)
    rules: list[Rule] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_header_and_ids(self) -> Self:
        # a missing effective date must be explained, never left silently blank.
        if self.effective_date is None and not (
            self.effective_date_note and self.effective_date_note.strip()
        ):
            raise ValueError("effective_date is null, so effective_date_note is required")
        ids = [rule.rule_id for rule in self.rules]
        duplicates = sorted({rule_id for rule_id in ids if ids.count(rule_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate rule_id(s): {duplicates}")
        # A section is either evaluated or out of scope, never both.
        out_of_scope = {note.section for note in self.out_of_scope_sections}
        clashes = sorted(out_of_scope & {rule.section for rule in self.rules})
        if clashes:
            raise ValueError(f"section(s) {clashes} have a rule but are declared out of scope")
        return self

    def covered_sections(self) -> set[str]:
        """Sections that have a rule, or are cited as a heading-only or referenced section."""
        return (
            {rule.section for rule in self.rules}
            | {note.section for note in self.heading_only_sections}
            | {note.section for note in self.referenced_sections}
        )
