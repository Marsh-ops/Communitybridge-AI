"""A deterministic check that the person's own words state the identity the model extracted.

Serves the Coordinator Agent's goal extraction. The design is that a language model interprets the goal and
a deterministic tool verifies what matters. Identity is the most sensitive field: the first live
measurements showed the model reading an APY town as "Aboriginal" when the person never said so
(SC-16, SC-18 and SC-26, and three held-out cases). The value feeds the §6.1.1 APY Lands exclusion
and the §6.1.2 Aboriginal Engagement Strategy benefits, so a wrong guess either excludes someone
who is eligible or grants a benefit nobody claimed. A missing value only costs one question, because
`not_stated` makes the plan ask (tools/entitlements.py). So the list below leans strict.

Why this is a list and not a PATS rule: the PDF says who is covered, not how a person words it, so
the wordings are a project decision. They are:

- **First person only, in the person's own words**: "I'm Aboriginal", "I am Torres Strait Islander",
  "I identify as Aboriginal and Torres Strait Islander". Nobody else's identity counts ("my mum is
  Aboriginal"), because identity is the person's own to declare; the plan asks instead.
- **`neither` needs both groups denied**: "I'm not Aboriginal or Torres Strait Islander", "I'm
  non-Indigenous". "I'm not Aboriginal" alone could still be a Torres Strait Islander.
- **Never inferred** from a place, a name, a job, a card, a nation or language-group name (Anangu,
  Kaurna), or a generic word ("Indigenous", "First Nations"): the three values cannot hold
  "Indigenous, which group", and mapping a name to a group is the system inferring identity.
- **A service or role name is not an identity**: "I'm an Aboriginal Health Practitioner" says what
  the person does, not who they are.
- **Hedged, "or" and "and/or" wordings are not statements**: "I think I'm...", "if I'm...", and
  the system's own question wording "Aboriginal and/or Torres Strait Islander" do not say which.

The check only ever removes a value. If the model's value is not the one the words state, it
becomes `not_stated`. It never adds one the model left out: that would make it a second extractor
with a second set of mistakes, and the cost of a missed statement is a single question at Gate 1.

The goal text is personal data. Nothing here writes it, or any value found in it, anywhere.
"""

import re
from dataclasses import dataclass

from communitybridge.models.request import IndigenousStatus

_ABORIGINAL = r"aboriginal\b"
# "Straight" and "Straits" are common spellings of Strait; the rest of the name must be exact.
_TORRES = r"torres\s+(?:strait|straight|straits)\s+islander\b"
_GROUP = rf"(?:{_ABORIGINAL}|{_TORRES})"
_ARTICLE = r"(?:(?:a|an)\s+)?"

# "I'm", "I am" or "I identify as", with the apostrophe optional (people type "Im" on a phone) and a
# few small words allowed in between ("I'm an", "I'm also a", "I'm proudly").
_AFFIRM_LEAD = r"\bi(?:'?m|\s+am|\s+identify\s+as)\s+(?:(?:a|an|also|proud|proudly|both)\s+)*"
# One group, optionally joined to a second: "Aboriginal and Torres Strait Islander".
_JOINER = r"(?:\s*,\s*and\s+|\s+and\s+|\s*&\s*|\s*,\s*|\s+as\s+well\s+as\s+)"
_AFFIRM = re.compile(
    rf"{_AFFIRM_LEAD}(?P<first>{_GROUP})(?:{_JOINER}{_ARTICLE}(?P<second>{_GROUP}))?"
)

# A denial has to name both groups; the lead is "I'm not", "I'm neither" or "I don't identify as".
_DENY_LEAD = r"\b(?:i(?:'?m|\s+am)\s+(?:not|neither)|i\s+(?:do\s+not|don'?t)\s+identify\s+as)\s+"
_DENY = re.compile(
    rf"{_DENY_LEAD}{_ARTICLE}(?P<first>{_GROUP})\s+(?:or|nor)\s+{_ARTICLE}(?P<second>{_GROUP})"
)
# "I'm non-Indigenous", "I'm not Indigenous", "I don't identify as Indigenous": a denial of the
# whole category, which covers both groups.
_DENY_INDIGENOUS = re.compile(
    r"\bi(?:'?m|\s+am)\s+(?:not\s+(?:an?\s+)?|non[-\s]?)indigenous\b"
    r"|\bi\s+(?:do\s+not|don'?t)\s+identify\s+as\s+(?:an?\s+)?indigenous\b"
)

# "I'm Aboriginal or Torres Strait Islander" and "...and/or..." do not say which one applies.
_UNDECIDED = re.compile(rf"\s*(?:and/or|or|nor|/|\+)\s*{_ARTICLE}{_GROUP}")
# The word right after the group, when the group is part of a service or role name: "Aboriginal
# Health Practitioner", "Aboriginal Liaison Officer", "Aboriginal Community Controlled...".
_NEXT_WORD = re.compile(r"\s+([a-z]+)")
_SERVICE_OR_ROLE_WORDS = frozenset(
    {
        "affairs",
        "center",
        "centre",
        "communities",
        "community",
        "controlled",
        "corp",
        "corporation",
        "council",
        "education",
        "engagement",
        "health",
        "hospital",
        "housing",
        "land",
        "lands",
        "language",
        "legal",
        "liaison",
        "medical",
        "officer",
        "organisation",
        "organization",
        "practitioner",
        "program",
        "programme",
        "service",
        "services",
        "strategy",
        "worker",
    }
)
# Words that, shortly before "I'm Aboriginal", make it a question or a guess and not a statement.
_HEDGES = frozenset(
    {
        "guess",
        "if",
        "maybe",
        "might",
        "perhaps",
        "possibly",
        "suppose",
        "think",
        "thought",
        "unsure",
        "whether",
        "wonder",
        "wondering",
    }
)
# Only the words just before the statement count: "I think I'm Aboriginal" is a guess, but "I want
# to know if I can bring my sister because I'm Aboriginal" is a statement with a question beside it.
_HEDGE_WINDOW = 3
# Reading back a fixed number of characters, not the whole sentence, keeps the time for a long goal
# proportional to its length: the text is whatever a person types, and a hostile or pasted one can
# repeat the statement thousands of times. Three words fit in far fewer characters than this.
_HEDGE_LOOKBACK_CHARS = 80

_GROUPS_TO_VALUE: dict[frozenset[str], IndigenousStatus] = {
    frozenset({"aboriginal"}): "aboriginal",
    frozenset({"torres_strait_islander"}): "torres_strait_islander",
    frozenset({"aboriginal", "torres_strait_islander"}): "both",
}


@dataclass(frozen=True)
class IdentityCheck:
    """The identity to use after the check, and whether the model's value had to be taken away."""

    value: IndigenousStatus
    removed: bool


def _sentences(text: str) -> list[str]:
    """Lower-case sentences with one kind of apostrophe and single spaces."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", " ".join(text.split())) if s]


def _group_name(matched: str) -> str:
    return "aboriginal" if matched.startswith("aboriginal") else "torres_strait_islander"


def _hedged(sentence: str, start: int) -> bool:
    """True if one of the few words just before the statement makes it a question or a guess."""
    nearby = sentence[max(0, start - _HEDGE_LOOKBACK_CHARS) : start]
    words = re.findall(r"[a-z']+", nearby)[-_HEDGE_WINDOW:]
    return any(word in _HEDGES for word in words)


def _is_service_or_role(after: str) -> bool:
    """True if the text right after a group name continues it into a service or role name."""
    following = _NEXT_WORD.match(after)
    return following is not None and following.group(1) in _SERVICE_OR_ROLE_WORDS


def stated_identity(goal_text: str) -> IndigenousStatus:
    """The identity the person's own words state, or `not_stated` if they do not state one.

    Reads every sentence. A person who names both groups across two sentences is `both`; one who
    both claims and denies an identity has stated nothing clear, so the answer is `not_stated`.
    """
    claimed: set[str] = set()
    denied = False
    for sentence in _sentences(goal_text):
        for found in _AFFIRM.finditer(sentence):
            after = sentence[found.end() :]
            if _hedged(sentence, found.start()) or _UNDECIDED.match(after):
                continue
            if _is_service_or_role(after):
                continue
            claimed.add(_group_name(found.group("first")))
            if found.group("second"):
                claimed.add(_group_name(found.group("second")))
        for found in _DENY.finditer(sentence):
            groups = {_group_name(found.group("first")), _group_name(found.group("second"))}
            if groups == {"aboriginal", "torres_strait_islander"} and not _hedged(
                sentence, found.start()
            ):
                denied = True
        for found in _DENY_INDIGENOUS.finditer(sentence):
            denied = denied or not _hedged(sentence, found.start())
    if denied:
        return "not_stated" if claimed else "neither"
    return _GROUPS_TO_VALUE.get(frozenset(claimed), "not_stated")


def check_identity(claimed: IndigenousStatus, goal_text: str) -> IdentityCheck:
    """Keep the model's identity only if the person's words state exactly that identity.

    Any other value, including one that is only partly right (the model says `both` and the person
    said one group), becomes `not_stated`, so Gate 1 shows it as not stated and the plan asks.
    """
    if claimed == "not_stated":
        return IdentityCheck(value="not_stated", removed=False)
    if stated_identity(goal_text) == claimed:
        return IdentityCheck(value=claimed, removed=False)
    return IdentityCheck(value="not_stated", removed=True)
