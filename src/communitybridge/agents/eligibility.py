"""The Eligibility Agent: gets the rules engine's decision and explains it.

The project design: this agent "explains the PATS decision in plain language"; its only data is
the rules engine's output; "the decision comes only from the rules engine"; and "the explanation
must cite the sections in the decision". The design diagram shows a model here, to explain. The core rule
says no language model output can set or change an eligibility result.

How the design keeps those promises (each is covered by a test):

- **The decision and the wording are different fields.** The node calls the rules engine, its one
  tool, and stores the whole `EligibilityResult` in `state.eligibility`. The model's text goes in
  `state.explanation`, whose schema (`EligibilityExplanation`) has no `eligible` field, so a model
  that tries to add one is refused like any other bad reply. The result is a frozen model, so
  nothing after it can edit it either.
- **A section the decision does not contain is refused.** `check_explanation` reads both the
  `cited_sections` list and the references in the text (`§6.4.6`, "section 6.4.6"), and a reply
  that cites any section outside the decision is sent back to the model with the section named.
  For an outcome that is not "eligible" the explanation must also cite a section behind it, and
  its wording must state the engine's outcome and not the opposite (a word check: "not eligible"
  for a client who is eligible is refused). Those checks prove the citations are real and the
  outcome is not reversed. They cannot prove every sentence is right, so the user is always
  shown the engine's outcome beside the text, never the model's version of it.
- **The engine's own words are always built, and are what the plan states as verified**. Every rule hit carries a plain-English `reason` written from the PDF, so the node builds
  `reasons` from them (`fallback_explanation`) on every run, whether or not a model is used. The
  plan checks and shows those as its statement. The model's wording is kept in `explanation`
  instead, apart from the plan, and shown to the user labelled as a language model's: its check
  covers the sections it cites and the outcome it states, not what it says about the client (a
  model once called an Aboriginal client "Aboriginal and Torres Strait Islander" and passed it).
- **If the model cannot be trusted, there is simply no model wording.** A model that is down, or
  that still fails after the two retries, is not an emergency: the decision and the reasons stand
  on their own, `explanation` stays empty, and the trace says why. This is the information-only
  agent in the table, so it never needs to end the run.
- **Only the rules engine can end the run here.** If the engine itself fails (a town or distance
  missing from the dataset, rules data it cannot use, an unreadable database) there is no decision
  to explain, so the run goes to staff with a fixed reason, never the exception text, which can
  hold the user's town. A bug (an unconfirmed goal, no service) raises instead.
- **The model sees only the result.** Its prompt holds the outcome and the rule results, and
  not the user's words, town or facts, so the trace and the model never need them.
"""

import re
from typing import Any, Literal

from communitybridge.graph.node_contract import NodeFn, traced_node
from communitybridge.graph.state import GraphState
from communitybridge.graph.toolbox import ToolBox
from communitybridge.llm.base import LLMError, LLMOutputError, LLMProvider
from communitybridge.llm.structured import generate
from communitybridge.models import EligibilityExplanation, TraceEvent
from communitybridge.models.engine import EligibilityOutcome, EligibilityResult
from communitybridge.models.llm import ChatMessage
from communitybridge.tools.distance import DistanceDataError, DistanceNotFoundError
from communitybridge.tools.rule_access import RuleParameterError
from communitybridge.tools.rules_engine import EngineInputError
from communitybridge.tools.service_query import ServiceQueryError

# The tool's name in the permission map (`AGENT_TOOLS` in graph/build.py) and in the trace.
RULES_ENGINE = "rules_engine"

OUTCOME_WORDS: dict[EligibilityOutcome, str] = {
    "eligible": "eligible",
    "not_eligible": "not eligible",
    "needs_review": "needs review by a PATS authorised officer",
}

SYSTEM_PROMPT = """\
You explain the result of a travel-subsidy rules check to a person in country South Australia who \
is planning to travel to see a medical specialist. The result below is final: you cannot change \
it, and you must never say or suggest anything different.

Rules:
- Use only the outcome and the listed rule results. Do not add facts, amounts, dates or \
conditions that are not in the list.
- Write 3 to 6 short, plain sentences. Say the outcome first, in these words: eligible, \
not eligible, or needs review. Use those words only for the overall outcome; for a single \
benefit say entitled or not entitled.
- Cite a section by writing § and its number exactly as listed, such as §6.4.6. Put every \
section you mention in cited_sections, and mention no section that is not listed.
- If the outcome is not eligible or needs review, say which listed rule results are the reason \
and cite them.
- A result marked not_applicable did not apply to this person; you need not mention it.
- Do not give medical advice. Do not promise that any payment will be made.
- Reply with the JSON object only.\
"""

# A reference to a section in free text: "§6.4.6", "section 6.4.6", and a list such as
# "sections 6.4.6 and 8.1.1" (the later numbers must be dotted, so "and 100 km" is not one).
_REFERENCE = re.compile(
    r"(?:§|\bsections?\b)\s*(\d+(?:\.\d+)*)((?:\s*(?:,|and|&|or)\s*§?\s*\d+(?:\.\d+)+)*)",
    re.IGNORECASE,
)
_DOTTED = re.compile(r"\d+(?:\.\d+)+")

# The words that must (and must not) appear for each outcome. A word check, not a proof: it
# catches an explanation that never states the outcome or states the opposite. "eligible" must
# not be negated, and a single benefit is "entitled" or "not entitled", never "not eligible".
_NEGATED = re.compile(r"\bnot eligible\b|\bineligible\b", re.IGNORECASE)
_ELIGIBLE = re.compile(r"\beligible\b", re.IGNORECASE)
_REVIEW = re.compile(r"\breview\b", re.IGNORECASE)


class EligibilityAgentError(Exception):
    """The graph asked this agent to do something it should never be asked to do (a bug)."""


def referenced_sections(text: str) -> set[str]:
    """Every section a piece of text refers to, as bare numbers such as "6.4.6"."""
    found: set[str] = set()
    for match in _REFERENCE.finditer(text):
        found.add(match.group(1))
        found.update(_DOTTED.findall(match.group(2)))
    return found


def _section_key(section: str) -> tuple[int, ...]:
    return tuple(int(part) for part in section.split("."))


def decision_sections(result: EligibilityResult) -> set[str]:
    """The sections an explanation may cite: those of the rule hits, and any a reason names.

    A reason can point to a section that is not a rule of its own, for example "a PATS authorised
    officer must review (§8.3)". That is still the decision's own text, so it can be cited.
    """
    sections = {hit.section for hit in result.decision.rule_hits}
    for hit in result.decision.rule_hits:
        sections |= referenced_sections(hit.reason)
    return sections


def _sections_behind(result: EligibilityResult) -> set[str]:
    return {h.section for h in result.decision.rule_hits if h.outcome in ("fail", "needs_review")}


def build_messages(result: EligibilityResult) -> list[ChatMessage]:
    """The request to the model: the instructions, then the result as data.

    Nothing about the user goes in: not their words, their town or their facts."""
    lines = [f"Outcome: {OUTCOME_WORDS[result.outcome]}", "Rule results:"]
    lines += [f"- §{h.section} [{h.outcome}] {h.reason}" for h in result.decision.rule_hits]
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content="\n".join(lines)),
    ]


def _outcome_problem(text: str, outcome: EligibilityOutcome) -> str | None:
    """A sentence if the text does not state the engine's outcome (or states the opposite)."""
    if outcome == "eligible":
        if _NEGATED.search(text):
            return (
                "the outcome is eligible, so the text must not say 'not eligible'; "
                "say 'not entitled' for a single benefit"
            )
        if not _ELIGIBLE.search(text):
            return "the text must say the outcome: eligible"
    elif outcome == "not_eligible":
        if not _NEGATED.search(text):
            return "the text must say the outcome: not eligible"
    elif not _REVIEW.search(text):
        return "the text must say the outcome: needs review"
    return None


def _clean(section: str) -> str:
    return section.strip().removeprefix("§").strip()


def check_explanation(explanation: EligibilityExplanation, result: EligibilityResult) -> str | None:
    """None if the explanation only cites sections in the decision, else one sentence of problems.

    The sentence is shown to the model and may reach the trace, so it names sections only.
    """
    allowed = decision_sections(result)
    cited = {_clean(s) for s in explanation.cited_sections}
    mentioned = referenced_sections(explanation.text)
    problems = []
    outside = sorted(cited - allowed, key=_section_key_or_text)
    if outside:
        problems.append(f"cited_sections lists {', '.join(outside)}, which is not in the decision")
    in_text = sorted(mentioned - allowed, key=_section_key_or_text)
    if in_text:
        problems.append(f"the text mentions {', '.join(in_text)}, which is not in the decision")
    unlisted = sorted((mentioned & allowed) - cited, key=_section_key)
    if unlisted:
        problems.append(
            f"the text mentions {', '.join(unlisted)} but cited_sections does not list it"
        )
    wording = _outcome_problem(explanation.text, result.outcome)
    if wording:
        problems.append(wording)
    behind = _sections_behind(result)
    if result.outcome != "eligible" and behind and not cited & behind:
        problems.append(
            "for this outcome cite at least one section behind it: "
            + ", ".join(sorted(behind, key=_section_key))
        )
    if not problems:
        return None
    allowed_text = ", ".join(sorted(allowed, key=_section_key))
    return "; ".join(problems) + f". Only these sections may be cited: {allowed_text}"


def _section_key_or_text(section: str) -> tuple[int, tuple[int, ...], str]:
    """Sort real section numbers as numbers, then anything the model made up as text.

    The leading 0 or 1 keeps the two kinds from ever being compared with each other: comparing
    an int with a str would raise, and one odd reply from the model must not crash the node."""
    try:
        return (0, _section_key(section), "")
    except ValueError:
        return (1, (), section)


def fallback_explanation(result: EligibilityResult) -> EligibilityExplanation:
    """An explanation built only from the rule reasons the engine already wrote from the PDF.

    Used when the model cannot be used or keeps failing the check. It says the outcome, then the
    reasons behind it first, then the others; rules that did not apply are left out. It adds
    nothing of its own, so it always agrees with the decision and always passes the same check.
    """
    hits = [h for h in result.decision.rule_hits if h.outcome != "not_applicable"]
    hits = hits or list(result.decision.rule_hits)
    # sorted() is stable, so the hits keep the engine's order within each group.
    hits = sorted(hits, key=lambda h: h.outcome not in ("fail", "needs_review"))
    text = " ".join([f"Outcome: {OUTCOME_WORDS[result.outcome]}.", *(h.reason for h in hits)])
    cited = {h.section for h in hits}
    for hit in hits:
        cited |= referenced_sections(hit.reason)
    return EligibilityExplanation(text=text, cited_sections=sorted(cited, key=_section_key))


def _event(task: str, source: str, result: str, status: Any) -> TraceEvent:
    return TraceEvent(
        task=task, agent="eligibility", tool_or_source=source, result=result, status=status
    )


def _escalate(reason: str) -> dict[str, Any]:
    """End the run for staff. Two events: what failed, and the hand-over."""
    return {
        "status": "escalated",
        "trace": [
            _event("decide eligibility", RULES_ENGINE, reason, "failed"),
            _event(
                "pass to staff",
                "escalation to staff",
                "no eligibility decision could be made; a staff member can take the request",
                "escalated",
            ),
        ],
    }


def _explain(
    provider: LLMProvider, result: EligibilityResult
) -> tuple[EligibilityExplanation | None, list[TraceEvent]]:
    """Ask the model to explain and check it. None if there is no wording the user can be shown.

    The engine's own reasons are built separately (the caller), so a failure here loses only the
    model's extra wording and never what the plan states.
    """
    source = f"llm:{provider.name}"
    try:
        generated = generate(
            provider,
            EligibilityExplanation,
            build_messages(result),
            check=lambda e: check_explanation(e, result),
        )
    except LLMOutputError as exc:
        # The last problem names sections and fields only (see check_explanation), never a value.
        problems = str(exc).partition("Last problems: ")[2] or "the reply did not pass its checks"
        why = f"the model's explanation was not accepted after {exc.attempts} calls: {problems}"
        status = "blocked"
    except LLMError:
        why = "the model could not be used (not running, or an unusable answer)"
        status = "failed"
    else:
        value = generated.value
        explanation = EligibilityExplanation(
            text=value.text,
            cited_sections=sorted({_clean(s) for s in value.cited_sections}, key=_section_key),
        )
        calls = f"{generated.attempts} LLM call" + ("" if generated.attempts == 1 else "s")
        summary = f"explanation written in {calls}; cites {', '.join(explanation.cited_sections)}"
        return explanation, [_event("explain decision", source, summary, "ok")]

    return None, [
        _event("explain decision", source, why, status),
        _event(
            "explain decision",
            "rules_engine reasons",
            "the plan states the rule reasons; there is no model wording to show",
            "ok",
        ),
    ]


def make_decide_eligibility(provider: LLMProvider, tools: ToolBox) -> NodeFn:
    """Build the decide_eligibility node around a provider and the tool box for this agent."""

    @traced_node("eligibility")
    def decide_eligibility(state: GraphState) -> dict[str, Any]:
        goal = state.goal
        if goal is None:
            raise EligibilityAgentError("the state has no goal to decide eligibility for")
        if not goal.confirmed_by_user:
            raise EligibilityAgentError("the goal has not been confirmed by the user (Gate 1)")
        if state.service is None:
            raise EligibilityAgentError("the state has no service to decide eligibility for")
        try:
            result: EligibilityResult = tools.call(
                RULES_ENGINE, goal, state.facts, state.service.service_id
            )
        except EngineInputError:
            return _escalate("the request does not match the dataset the rules engine uses")
        except DistanceNotFoundError:
            return _escalate("the dataset has no road distance for this journey")
        except (DistanceDataError, RuleParameterError):
            return _escalate("the rules or distance data could not be used")
        except ServiceQueryError:
            return _escalate("the service database could not be read")

        decided = _event(
            "decide eligibility",
            RULES_ENGINE,
            f"outcome: {OUTCOME_WORDS[result.outcome]}; {len(result.decision.rule_hits)} rule hits "
            f"from PATS V{result.decision.rules_version}",
            "ok",
        )
        explanation, events = _explain(provider, result)
        return {
            "eligibility": result,
            "reasons": fallback_explanation(result),  # built by the rules, on every path
            "explanation": explanation,  # the model's wording, or None
            "trace": [decided, *events],
        }

    return decide_eligibility


def route_after_eligibility(state: GraphState) -> Literal["plan", "inform", "finish"]:
    """An eligible client goes on to Logistics. One who is not eligible, or whose case needs
    review, gets the information-only plan. An escalated run finishes."""
    if state.status == "escalated":
        return "finish"
    if state.eligibility is None:
        raise EligibilityAgentError("eligibility ended with no result and was not escalated")
    return "plan" if state.eligibility.outcome == "eligible" else "inform"
