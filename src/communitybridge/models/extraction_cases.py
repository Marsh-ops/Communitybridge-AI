"""Schema for the held-out goal-extraction cases (data/scenarios/extraction_holdout.yaml;).

The 30 scenarios are the project's test set for goal extraction, and the first live measurement showed which of them fail and why. Any prompt change made after that is shaped by those
failures, so a score on the same 30 is optimistic. These cases are fresh goal texts, written after
that finding but before the changed pipeline was run on them, so they give a fairer second reading.
They check extraction only: unlike a scenario, a case carries no eligibility outcome.

`kept_as_written` lists the fields where the right answer is the user's own words, because the
name is not in the project's lists (a town or specialty the dataset does not know). Every other
town and specialty must be a listed name, which a test checks against the vocabulary.
"""

from datetime import date
from typing import Literal

from pydantic import Field

from communitybridge.models.common import NonEmptyStr, StrictModel
from communitybridge.models.request import GoalRequest


class ExtractionCase(StrictModel):
    id: str = Field(pattern=r"^HO-\d{2}$")
    goal: NonEmptyStr
    expected_fields: GoalRequest
    kept_as_written: list[Literal["origin_town", "specialty"]] = Field(default_factory=list)
    synthetic: Literal[True]


class ExtractionCaseSet(StrictModel):
    reference_date: date  # "next month" and similar are resolved against this date
    conventions: list[NonEmptyStr] = Field(min_length=1)
    cases: list[ExtractionCase] = Field(min_length=1)
