"""Entitlement rules: the Aboriginal Engagement Strategy, advances and escorts.

Part of the deterministic rules engine. Once a client is eligible, this tool says
what they may *receive*:
- the Aboriginal Engagement Strategy benefits, which apply to Aboriginal and Torres Strait
  Islander clients: an advance on subsidies, no concession card needed for the first night's
  accommodation subsidy, and automatic approval for an escort (§6.1.2);
- an advance on travel subsidies for everyone else, if they hold a concession card, are in severe
  financial hardship and received under five business days' notice (§6.5.5);
- an escort: one only, aged 18 or over. It is automatic for Aboriginal and Torres Strait Islander
  clients and for children aged 17 or under; otherwise the treating specialist must approve it
  under one of the clinical criteria (§7.2.8, §7.2.9).

Design choices (the project plan stores the rules as versioned data):
- Ages, notice limits and who is covered all come from data/rules/pats_v26_4.yaml, and a missing
  entry makes the tool raise `RuleParameterError` instead of quietly granting or refusing.
- Not having an entitlement is information, not a failure: the client stays eligible for
  everything else. So hits are `pass` (granted) or `not_applicable` (not granted), never `fail`.
- Interpretations: an Aboriginal client's escort is automatic, so §7.2.9 approval is only for the
  other criteria; the strategy waives the three §6.5.5 tests; and where the answer depends on an
  identity the client has not stated, the client is asked and the decision is not blocked.
- The first-night accommodation subsidy itself (§7.3.1) is worked out with the other costs in the
  logistics calculator. This tool only reports the §6.1.2 waiver of the concession card.
"""

from typing import Literal

from communitybridge.models.eligibility import RuleHit
from communitybridge.models.entitlements import (
    EntitlementAssessment,
    EntitlementInput,
    EntitlementStatus,
    EscortStatus,
)
from communitybridge.models.rules import Rule, RuleSet
from communitybridge.tools.rule_access import (
    RuleParameterError,
    entry_with,
    find_rule,
    make_hit,
    norm,
    number,
    require,
    text_list,
)

AES_ADVANCE_RULE = "PATS-6.1.2-ADVANCE"
AES_FIRST_NIGHT_RULE = "PATS-6.1.2-FIRST-NIGHT"
AES_ESCORT_RULE = "PATS-6.1.2-ESCORT"
ADVANCE_RULE = "PATS-6.5.5-ADVANCE"
ESCORT_RULE = "PATS-7.2.8-ESCORT"
ESCORT_AUTH_RULE = "PATS-7.2.9-ESCORT-AUTH"

IDENTITY_QUESTION = "Do you identify as Aboriginal and/or Torres Strait Islander?"
NOTICE_QUESTION = "How many business days' notice did you receive of the appointment?"

_Identity = Literal["applies", "no", "ask"]
_FROM_IDENTITY: dict[_Identity, EntitlementStatus] = {
    "applies": "entitled",
    "no": "not_entitled",
    "ask": "ask",
}

_ASK_REASON = (
    "This depends on whether the client identifies as Aboriginal and/or Torres Strait Islander, "
    "and they have not said, so the plan asks."
)


def _identity(identity: str, rule: Rule, list_name: str) -> _Identity:
    """Does the strategy (or an automatic route) cover this client? `ask` if they haven't said."""
    # "ask" is the only unstated-identity outcome the code implements here.
    require(rule, "not_stated_outcome", "ask")
    if identity == "not_stated":
        return "ask"
    covered = {norm(status) for status in text_list(rule, list_name)}
    return "applies" if norm(identity) in covered else "no"


def _strategy_hit(rule: Rule, status: _Identity, benefit: str) -> RuleHit:
    """One §6.1.2 benefit: granted, not for this client, or depends on the client's answer."""
    if status == "applies":
        return make_hit(rule, "pass", f"{benefit} (Aboriginal Engagement Strategy, §6.1.2).")
    if status == "ask":
        return make_hit(rule, "not_applicable", _ASK_REASON)
    return make_hit(
        rule,
        "not_applicable",
        "The Aboriginal Engagement Strategy does not apply, as the client is not Aboriginal or "
        "Torres Strait Islander (§6.1.2).",
    )


# --- §6.5.5 advance ------------------------------------------------------------------------------


def _assess_advance(
    request: EntitlementInput, rules: RuleSet, strategy: _Identity
) -> tuple[EntitlementStatus, RuleHit, list[str]]:
    rule = find_rule(rules, ADVANCE_RULE)
    criteria = text_list(rule, "criteria_all_required")
    card = entry_with(criteria, "concession card", rule, "criteria_all_required")
    hardship = entry_with(criteria, "hardship", rule, "criteria_all_required")
    notice = entry_with(criteria, "notice", rule, "criteria_all_required")
    limit = number(rule, "notice_business_days_less_than")
    tests = f"{card}; {hardship}; {notice}"

    if strategy == "applies":
        # the strategy waives all three tests.
        return (
            "entitled",
            make_hit(
                rule,
                "not_applicable",
                f"The §6.5.5 tests ({tests}) are waived for this client under the Aboriginal "
                "Engagement Strategy (§6.1.2).",
            ),
            [],
        )

    unmet = []
    if not request.concession_card:
        unmet.append(card)
    if not request.severe_financial_hardship:
        unmet.append(hardship)
    notice_known = request.notice_business_days is not None
    if notice_known and request.notice_business_days >= limit:  # "less than", so 5 is not enough
        unmet.append(notice)

    if unmet:
        text = f"An advance needs all of: {tests}. Not met: {'; '.join(unmet)}."
        if strategy == "ask":
            return "ask", make_hit(rule, "not_applicable", f"{text} {_ASK_REASON}"), []
        return "not_entitled", make_hit(rule, "not_applicable", text), []
    if not notice_known:
        return (
            "ask",
            make_hit(
                rule,
                "not_applicable",
                f"The client holds a concession card and is in severe financial hardship, but the "
                f"notice they received is not known: {notice}.",
            ),
            [NOTICE_QUESTION],
        )
    return (
        "entitled",
        make_hit(rule, "pass", f"The client meets all three criteria for an advance ({tests})."),
        [],
    )


# --- §7.2.8 and §7.2.9 escort ----------------------------------------------------------------


def _non_automatic_criterion(rule: Rule, criterion: str) -> str:
    """The §7.2.9 criterion as the data names it; refuse if the data no longer lists it."""
    listed = text_list(rule, "criteria")
    automatic = {norm(c) for c in text_list(rule, "automatic_criteria")}
    for entry in listed:
        if norm(entry) == norm(criterion) and norm(entry) not in automatic:
            return entry
    raise RuleParameterError(
        f"{rule.rule_id}: criterion {criterion!r} is not a non-automatic criterion in the data"
    )


def _assess_escort(
    request: EntitlementInput, rules: RuleSet
) -> tuple[EscortStatus, list[RuleHit], list[str]]:
    if not request.escort_requested:
        return "not_requested", [], []

    aes_rule = find_rule(rules, AES_ESCORT_RULE)
    rule = find_rule(rules, ESCORT_RULE)
    auth = find_rule(rules, ESCORT_AUTH_RULE)

    strategy = _identity(request.indigenous_status, aes_rule, "applies_to_indigenous_status")
    automatic = _identity(request.indigenous_status, rule, "automatic_for_indigenous_status")
    max_child_age = int(number(rule, "automatic_child_max_age"))
    min_escort_age = int(number(rule, "escort_min_age"))
    max_escorts = int(number(rule, "max_escorts_per_client"))
    by_identity = "applies" in (strategy, automatic)
    by_age = request.client_age <= max_child_age
    by_approval = request.escort_criterion is not None and request.escort_specialist_approved

    hits = [_strategy_hit(aes_rule, strategy, "Automatic approval for an escort")]
    # The PDF says "one escort only"; spell the common case the way it does.
    count = {1: "one escort"}.get(max_escorts, f"{max_escorts} escorts")
    limits = f" Each client is entitled to {count} only, aged {min_escort_age} or older (§7.2.8)."

    if by_identity or by_age or by_approval:
        routes = []
        if by_identity:
            routes.append(
                "Aboriginal and Torres Strait Islander clients are automatically entitled"
            )
        if by_age:
            routes.append(f"children aged {max_child_age} or under are automatically entitled")
        approval_hit = None
        if by_approval:
            criterion = _non_automatic_criterion(auth, request.escort_criterion)
            routes.append("the treating medical specialist has approved the escort")
            approval_hit = make_hit(
                auth,
                "pass",
                f"The treating medical specialist has approved an escort under '{criterion}' "
                "(§7.2.9).",
            )
        sentence = "; ".join(routes)
        hits.append(make_hit(rule, "pass", f"{sentence[:1].upper()}{sentence[1:]}.{limits}"))
        if approval_hit is not None:
            hits.append(approval_hit)
        return "entitled", hits, []

    # Not automatic. The only other route is the treating specialist's approval (§7.2.9).
    if request.escort_criterion is not None:
        criterion = _non_automatic_criterion(auth, request.escort_criterion)
        approval = (
            f"An escort under '{criterion}' needs the approval of the treating medical specialist "
            "(§7.2.9), and that has not been approved."
        )
    else:
        approval = (
            "Otherwise an escort needs the approval of the treating medical specialist under one "
            "of the clinical criteria (§7.2.9), and none has been given."
        )
    if strategy == "ask" or automatic == "ask":
        hits.append(make_hit(rule, "not_applicable", _ASK_REASON))
        hits.append(make_hit(auth, "not_applicable", approval))
        return "ask", hits, [IDENTITY_QUESTION]

    hits.append(
        make_hit(
            rule,
            "not_applicable",
            f"The client is not automatically entitled to an escort (not Aboriginal or Torres "
            f"Strait Islander, and over {max_child_age}).{limits}",
        )
    )
    hits.append(make_hit(auth, "not_applicable", approval))
    return "not_entitled", hits, []


# --- All the entitlements -------------------------------------------------------------------------


def assess_entitlements(request: EntitlementInput, rules: RuleSet) -> EntitlementAssessment:
    """What the client may receive: an advance, the first-night concession waiver and an escort.

    Sections §6.1.2, §6.5.5, §7.2.8 and §7.2.9 (see the module note). If the answer depends on an
    identity the client has not stated, it adds a question to `questions` instead of guessing. The hits are `pass` or `not_applicable`, never `fail`.
    """
    advance_rule = find_rule(rules, AES_ADVANCE_RULE)
    first_night_rule = find_rule(rules, AES_FIRST_NIGHT_RULE)
    status = request.indigenous_status

    advance_strategy = _identity(status, advance_rule, "applies_to_indigenous_status")
    first_night_strategy = _identity(status, first_night_rule, "applies_to_indigenous_status")

    advance, advance_hit, advance_questions = _assess_advance(request, rules, advance_strategy)
    escort, escort_hits, escort_questions = _assess_escort(request, rules)

    hits = [
        _strategy_hit(advance_rule, advance_strategy, "Subsidies can be paid in advance"),
        _strategy_hit(
            first_night_rule,
            first_night_strategy,
            "A concession card is not needed for the first night's accommodation subsidy",
        ),
        advance_hit,
        *escort_hits,
    ]
    first_night = _FROM_IDENTITY[first_night_strategy]

    asked = [*advance_questions, *escort_questions]
    if status == "not_stated":
        asked.insert(0, IDENTITY_QUESTION)
    questions = list(dict.fromkeys(asked))  # no repeats, order kept
    return EntitlementAssessment(
        escort=escort,
        advance=advance,
        first_night_concession_waived=first_night,
        questions=questions,
        rule_hits=hits,
    )
