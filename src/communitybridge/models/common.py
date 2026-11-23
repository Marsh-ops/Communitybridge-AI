"""Shared base model and the Citation type used by every agent.

Every boundary model inherits `StrictModel`, which forbids unknown fields. This matters most
for LLM output: if the model invents a field (for example `eligible` on an explanation),
validation fails instead of silently dropping or accepting it. That is how the original design
("language models interpret and explain; deterministic tools decide") is enforced at the
data level rather than by trusting the prompt.
"""

from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

# A string that must contain something other than whitespace (whitespace is stripped first).
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# A PATS section number exactly as numbered in the PDF, e.g. "6.4.6" (no trailing dot or "§").
SECTION_PATTERN = r"^\d+(\.\d+)*$"


class StrictModel(BaseModel):
    """Base for all boundary models: unknown fields are an error, not ignored."""

    model_config = ConfigDict(extra="forbid")


class Citation(StrictModel):
    """Where a factual claim came from. The Verifier checks `last_verified` for freshness."""

    source: NonEmptyStr
    section_or_record: NonEmptyStr
    last_verified: date
