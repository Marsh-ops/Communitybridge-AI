"""The closed lists of specialty and town names, and the check on the model's pick.

Serves the Coordinator Agent's goal extraction. The first live measurement showed people
say "heart specialist" or "Pukatja" while the rules engine and the distance tool need "Cardiology"
and "Pukatja (Ernabella)": 21 of the 37 wrong fields were that mismatch. The fix keeps the original
design, in which a language model interprets and a deterministic tool verifies:

1. the model is shown the project's own lists and asked to choose a name from them;
2. this tool then checks the choice. A match gets the listed spelling. A name with no match is kept
   exactly as the person wrote it, and the field is noted, so the user sees it at Gate 1 and a later
   tool can refuse it with a clear reason, instead of the system guessing.

The lists are built from data the project already holds and nothing new is invented: the PATS
section 9.2 accepted specialities in the rules file (encoded from the PDF), the specialty names the
services dataset uses (three of which, Physiotherapy, chemotherapy and renal dialysis, are not on
the PATS list), and the dataset's towns. Matching ignores capitals and extra spaces and nothing
else: "Pukatja" is not taken as "Pukatja (Ernabella)", because a guess by prefix could pick the
wrong place.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from pydantic import ConfigDict, model_validator

from communitybridge.db.seed import load_services, load_towns
from communitybridge.models import ExtractedGoal, StrictModel
from communitybridge.models.common import NonEmptyStr
from communitybridge.models.dataset import ServiceRecord, Town
from communitybridge.models.rules import RuleSet
from communitybridge.tools.rule_access import find_rule, norm, text_list
from communitybridge.tools.rules_loader import load_rules

SPECIALITIES_RULE_ID = "PATS-9.2-SPECIALITIES"


class Vocabulary(StrictModel):
    """The names the Coordinator may choose from. Frozen: it is reference data, not state."""

    model_config = ConfigDict(frozen=True)

    specialties: tuple[NonEmptyStr, ...]
    towns: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def _lists_are_not_empty_and_have_no_repeats(self) -> Self:
        for label, names in (("specialties", self.specialties), ("towns", self.towns)):
            if not names:
                raise ValueError(f"{label} must not be empty")
            if len({norm(name) for name in names}) != len(names):
                raise ValueError(f"{label} lists a name more than once")
        return self


def _first_of_each(names: Sequence[str]) -> tuple[str, ...]:
    """Drop repeats that differ only in capitals or spaces; the first spelling is kept."""
    seen: set[str] = set()
    kept: list[str] = []
    for name in names:
        if norm(name) not in seen:
            seen.add(norm(name))
            kept.append(name)
    return tuple(kept)


def load_vocabulary(
    rules: RuleSet | None = None,
    towns: Sequence[Town] | None = None,
    services: Sequence[ServiceRecord] | None = None,
) -> Vocabulary:
    """Build the lists from the rules file and the dataset. Arguments are for tests only.

    The PATS names come first, so where the dataset spells a name differently (only in capitals) the
    PATS spelling wins. A rules file with no section 9.2 rule is a named `RuleParameterError`,
    never an empty list that would quietly accept nothing.
    """
    rules = load_rules() if rules is None else rules
    accepted = text_list(find_rule(rules, SPECIALITIES_RULE_ID), "accepted_specialities")
    dataset_services = load_services() if services is None else services
    dataset_towns = load_towns() if towns is None else towns
    return Vocabulary(
        specialties=_first_of_each([*accepted, *(s.specialty for s in dataset_services)]),
        towns=_first_of_each([t.name for t in dataset_towns]),
    )


def canonical_name(name: str, names: Sequence[str]) -> str | None:
    """The listed spelling of `name`, or None if it is not on the list."""
    wanted = norm(name)
    if not wanted:
        return None
    for listed in names:
        if norm(listed) == wanted:
            return listed
    return None


@dataclass(frozen=True)
class Resolution:
    goal: ExtractedGoal
    # One note per field kept as written. Field names only: the user's words are personal data and
    # these notes are written into the trace.
    notes: tuple[str, ...]


def resolve_goal(goal: ExtractedGoal, vocabulary: Vocabulary) -> Resolution:
    """Check the model's town and specialty against the lists. Other fields are never touched."""
    updates: dict[str, str] = {}
    notes: list[str] = []
    for field, names in (("origin_town", vocabulary.towns), ("specialty", vocabulary.specialties)):
        listed = canonical_name(getattr(goal, field), names)
        if listed is None:
            notes.append(f"{field} kept as written (not in the known list)")
        else:
            updates[field] = listed
    resolved = ExtractedGoal.model_validate({**goal.model_dump(), **updates})
    return Resolution(goal=resolved, notes=tuple(notes))
