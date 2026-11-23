"""Verifier tool: citation, freshness and conflict checks on plan claims.

The Verifier Agent's only tool: "challenges stale, unsupported or conflicting
claims", with a "deterministic citation, freshness and conflict check", and "escalates uncertainty
to the user". It never calls a language model, so each block is a rule applied to data and can be
tested (the core rule: no uncited statements in a plan; a blocked one is shown as blocked).

A claim passes only if all three checks pass:

1. **Citation.** It must cite at least one source, and every source must be one the system can
   check: a PATS section that really exists in the guideline data (so an invented section number
   is caught), of the guideline version in use, or a record in the dataset.
2. **Freshness.** Every source must have been verified no longer ago than `MAX_SOURCE_AGE_DAYS`
   (the setting in `config.py`). A source dated in the future is rejected too.
3. **Conflict.** Two claims about the same subject that state different values both fail. Only
   claims that passed the first two checks are compared, so a claim with no source or an old one
   can't veto a well-sourced one.

Stale and conflicting information is also listed as an *escalation*: the user is asked to confirm
it, rather than the system choosing silently. If a *core* claim (one the whole plan rests on) is
blocked, the report says the plan must be re-planned (the design diagram).

The checks work on the claims and the catalogue passed in and change nothing (the Verifier is
read-only on every source).
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from communitybridge.config import ConfigError, max_source_age_days
from communitybridge.models.common import Citation
from communitybridge.models.plan import PlanItem
from communitybridge.models.rules import RuleSet
from communitybridge.models.verification import Claim, ClaimResult, Finding, VerificationReport
from communitybridge.tools.rule_access import norm
from communitybridge.tools.rules_engine import EngineContext

# A citation whose source starts with this is a citation of the PATS guideline.
PATS_SOURCE = "SA Health PATS Guidelines for Assessment"


@dataclass(frozen=True)
class SourceCatalogue:
    """Everything the Verifier will accept as a source. Built from the data, never from a model."""

    pats_label: str  # e.g. "SA Health PATS Guidelines for Assessment V26.4"
    pats_sections: frozenset[str]  # section numbers in the guideline data, e.g. "6.4.6"
    records: frozenset[str]  # ids of dataset records, e.g. "SVC-ADL-CARD-01"


def build_catalogue(rules: RuleSet, record_ids: Iterable[str]) -> SourceCatalogue:
    """The guideline sections the rules data knows, plus the record ids the caller supplies."""
    sections = rules.covered_sections()
    sections |= {note.section for note in rules.assumed_preconditions}
    sections |= {note.section for note in rules.out_of_scope_sections}
    return SourceCatalogue(
        pats_label=f"{PATS_SOURCE} V{rules.guideline_version}",
        pats_sections=frozenset(sections),
        records=frozenset(record_ids),
    )


def catalogue_from_context(
    context: EngineContext, extra_record_ids: Iterable[str] = ()
) -> SourceCatalogue:
    """The catalogue for the engine's data: every service and town, plus any extra record ids."""
    records = {service.service_id for service in context.services}
    records |= {town.town_id for town in context.towns.values()}
    records |= set(extra_record_ids)
    return build_catalogue(context.rules, records)


def citation_for_section(section: str, rules: RuleSet) -> Citation:
    """A citation of a PATS section, dated by when the rules data was last checked against the PDF.

    It makes no promise that the section exists; the Verifier checks that.
    """
    return Citation(
        source=f"{PATS_SOURCE} V{rules.guideline_version}",
        section_or_record=f"§{section}",
        last_verified=rules.last_verified,
    )


# --- Reading a citation ------------------------------


def _section_of(text: str) -> str | None:
    """'§6.4.6', 'Section 6.4.6' and '6.4.6 p. 8' all mean section 6.4.6."""
    match = re.search(r"\d+(?:\.\d+)*", text)
    return match.group(0) if match else None


def _record_of(text: str) -> str:
    """'service SVC-ADL-CARD-01' and 'SVC-ADL-CARD-01' both name the record SVC-ADL-CARD-01."""
    words = text.split()
    return words[-1] if words else text


def _is_pats(citation: Citation) -> bool:
    return citation.source.startswith(PATS_SOURCE)


def _describe(citation: Citation) -> str:
    """How the user should see the source."""
    if _is_pats(citation):
        return f"PATS {citation.section_or_record}"
    return _record_of(citation.section_or_record)


# --- The three checks ------------------------------


def _citation_findings(claim: Claim, catalogue: SourceCatalogue) -> list[Finding]:
    if not claim.citations:
        return [
            Finding(
                check="citation",
                detail="This statement has no source, so it can't be shown as fact.",
            )
        ]
    findings = []
    for citation in claim.citations:
        detail = None
        if _is_pats(citation):
            section = _section_of(citation.section_or_record)
            if citation.source != catalogue.pats_label:
                detail = (
                    f"It cites '{citation.source}', but the guideline in use is "
                    f"'{catalogue.pats_label}'."
                )
            elif section not in catalogue.pats_sections:
                shown = section or citation.section_or_record
                detail = f"It cites PATS §{shown}, which isn't in the guideline data."
        elif _record_of(citation.section_or_record) not in catalogue.records:
            detail = (
                f"Its source '{citation.source}' ({citation.section_or_record}) isn't one "
                "this system can check."
            )
        if detail:
            findings.append(Finding(check="citation", detail=detail))
    return findings


def _freshness_findings(
    claim: Claim, today: date, max_age_days: int
) -> tuple[list[Finding], list[str]]:
    """Findings for sources that are too old (or impossibly new), and what to ask the user."""
    findings: list[Finding] = []
    questions: list[str] = []
    for citation in claim.citations:
        age = (today - citation.last_verified).days
        what = _describe(citation)
        if age < 0:
            findings.append(
                Finding(
                    check="freshness",
                    detail=(
                        f"A source ({what}) is dated in the future ({citation.last_verified}), "
                        "which can't be right."
                    ),
                )
            )
            questions.append(
                f"The date on the source for {what} is in the future. Please check that record."
            )
        elif age > max_age_days:
            findings.append(
                Finding(
                    check="freshness",
                    detail=(
                        f"It relies on {what}, last checked {age} days ago "
                        f"({citation.last_verified}), which is more than the {max_age_days} days "
                        "allowed for current information."
                    ),
                )
            )
            questions.append(
                f"The information about {what} was last checked on {citation.last_verified} "
                f"({age} days ago). Please confirm it is still current."
            )
    return findings, questions


def _conflicts(claims: list[Claim], passed: list[bool]) -> tuple[dict[int, Finding], list[str]]:
    """Claims on one subject that state different values; only claims that passed are compared."""
    by_subject: dict[str, list[int]] = {}
    for index, claim in enumerate(claims):
        if passed[index] and claim.subject is not None:
            by_subject.setdefault(norm(claim.subject), []).append(index)

    found: dict[int, Finding] = {}
    questions: list[str] = []
    for members in by_subject.values():
        distinct: dict[str, str] = {}  # normalised value -> value as written, in order seen
        for index in members:
            value = claims[index].value or ""
            distinct.setdefault(norm(value), value)
        if len(distinct) < 2:
            continue
        subject = claims[members[0]].subject or ""
        for index in members:
            mine = norm(claims[index].value or "")
            others = ", ".join(f"'{v}'" for key, v in distinct.items() if key != mine)
            found[index] = Finding(
                check="conflict",
                detail=(
                    f"It conflicts with another statement about {subject} ({others}), so "
                    "neither is shown until it is confirmed."
                ),
            )
        values = " and ".join(f"'{v}'" for v in distinct.values())
        questions.append(
            f"Two sources disagree about {subject}: {values}. Please confirm which is right."
        )
    return found, questions


# --- The verdict ------------------------------


def verify_claims(
    claims: list[Claim],
    catalogue: SourceCatalogue,
    today: date | None = None,
    max_age_days: int | None = None,
) -> VerificationReport:
    """Check every claim and return the plan items: verified, or blocked with the reason why.

    `today` and `max_age_days` default to the real date and the `MAX_SOURCE_AGE_DAYS` setting;
    passing them makes a check repeatable (the tests do).
    """
    checked_on = today if today is not None else date.today()
    limit = max_age_days if max_age_days is not None else max_source_age_days()
    if limit < 1:
        raise ConfigError(f"the age limit must be at least 1 day, got {limit}")

    findings: list[list[Finding]] = []
    questions: list[str] = []
    for claim in claims:
        found = _citation_findings(claim, catalogue)
        stale, asks = _freshness_findings(claim, checked_on, limit)
        findings.append([*found, *stale])
        questions.extend(asks)

    conflicts, asks = _conflicts(claims, [not f for f in findings])
    for index, finding in conflicts.items():
        findings[index].append(finding)
    questions.extend(asks)

    results = []
    for claim, claim_findings in zip(claims, findings, strict=True):
        if claim_findings:
            item = PlanItem(
                statement=claim.statement,
                citations=claim.citations,
                status="blocked",
                blocked_reason=" ".join(f.detail for f in claim_findings),
            )
        else:
            item = PlanItem(statement=claim.statement, citations=claim.citations, status="verified")
        results.append(ClaimResult(item=item, findings=claim_findings, core=claim.core))
    return VerificationReport(
        results=results,
        escalations=list(dict.fromkeys(questions)),
        checked_on=checked_on,
        max_age_days=limit,
    )
