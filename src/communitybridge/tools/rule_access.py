"""Helpers for reading PATS rules from data, shared by the deterministic tools.

Every tool that applies a PATS rule (distance, the rules-engine gates, later the logistics
calculator) needs the same four things: find a rule by id, read a parameter, read a number, and
check that the data still says what the code implements. Keeping them in one place means all the
tools fail the same way and with the same named error, `RuleParameterError`.

Why the "require" check exists (Design: rules are versioned data, so a guideline update changes the
YAML and not the code): the YAML holds the numbers and lists, but the *meaning* of a test, such as
"greater than" or "all criteria required", is code. If a guideline update changes the meaning in
the YAML, the tool must stop, not quietly apply the old logic.
"""

import re
from typing import Any

from communitybridge.models.distance import OutOfScopeNotice
from communitybridge.models.eligibility import RuleHit, RuleOutcome
from communitybridge.models.rules import Rule, RuleSet


class RuleParameterError(Exception):
    """A rule is missing from the rules data, or a parameter is missing, malformed or not one the
    code implements."""


def find_rule(rules: RuleSet, rule_id: str) -> Rule:
    for rule in rules.rules:
        if rule.rule_id == rule_id:
            return rule
    raise RuleParameterError(f"rule {rule_id} is missing from the rules data")


def param(rule: Rule, name: str) -> Any:
    try:
        return rule.parameters[name]
    except KeyError:
        raise RuleParameterError(f"{rule.rule_id} has no parameter {name!r}") from None


def number(rule: Rule, name: str) -> float:
    value = param(rule, name)
    # bool is a subclass of int in Python, so exclude it explicitly: `true` is not a threshold.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuleParameterError(
            f"{rule.rule_id}: parameter {name!r} must be a number, got {value!r}"
        )
    return float(value)


def text_list(rule: Rule, name: str) -> list[str]:
    value = param(rule, name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuleParameterError(
            f"{rule.rule_id}: parameter {name!r} must be a list of text, got {value!r}"
        )
    return value


def require(rule: Rule, name: str, expected: Any) -> None:
    """The code implements one reading of the rule; refuse data that says something else."""
    actual = param(rule, name)
    if actual != expected:
        raise RuleParameterError(
            f"{rule.rule_id}: {name} is {actual!r}, but this tool only implements {expected!r}"
        )


def make_hit(rule: Rule, outcome: RuleOutcome, reason: str) -> RuleHit:
    """A RuleHit that cites the rule's own id and section, so a hit can't cite the wrong one."""
    return RuleHit(rule_id=rule.rule_id, section=rule.section, outcome=outcome, reason=reason)


def norm(text: str) -> str:
    """Compare text ignoring case and runs of spaces, so 'Cardiology ' matches 'cardiology'."""
    return " ".join(text.casefold().split())


def words(text: str) -> str:
    """Compare text as plain lowercase words, ignoring punctuation: 'non-urgent' == 'non urgent'.

    Used to match a closed-list value such as `bypassing_waiting_lists` against the PDF's own
    sentence in the rules data.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def entry_with(entries: list[str], keyword: str, rule: Rule, list_name: str) -> str:
    """The entry in a rules list that mentions `keyword`; refuse if the data has no such entry.

    A missing entry must never mean "no problem found", so this raises instead of returning None.
    """
    for entry in entries:
        if keyword in norm(entry):
            return entry
    raise RuleParameterError(f"{rule.rule_id}: {list_name} has no entry that mentions {keyword!r}")


def out_of_scope_notices(rules: RuleSet, shown_when: str) -> list[OutOfScopeNotice]:
    """The "this prototype doesn't assess section X" notices that apply in this situation.

    §6.5.1 (Cumulative Travel) is shown when the distance requirement is not met, and
    §6.4.7 (Life-limiting illness) when the nearest-service requirement is not met,
    so a client who might qualify under them is pointed to a person, not just told "not eligible".
    """
    return [
        OutOfScopeNotice(
            section=note.section,
            title=note.title,
            message=note.user_message,
            source_page=note.source_page,
        )
        for note in rules.out_of_scope_sections
        if note.shown_when == shown_when
    ]
