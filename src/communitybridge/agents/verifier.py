"""The Verifier Agent: checks the plan's statements before the user sees it.

The project design: the Verifier "challenges stale, unsupported or conflicting claims", reads all
agent outputs and sources (read-only), and runs a "deterministic citation, freshness and conflict
check". It "escalates uncertainty to the user". The design diagram shows it without a model, so unlike
Service Discovery and Logistics this agent is exactly as originally designed. In the graph it
sits between the agents that produce results and the user's approval of the plan (Gate 2), so a
statement with no source can never reach the user as if it were fact.

Design points worth being able to explain:

- **It is read-only.** It reads the service, the decision, the engine's reasons, the logistics
  result and the options from the state, and writes only the plan (the checked statements), the
  questions for the user, and whether the plan has to go back. It cannot edit a decision, an option
  or a cost.
- **It never reads the language model's wording**. The statement about the
  decision is the rules engine's own reasons (`state.reasons`). The model's wording
  (`state.explanation`) is shown to the user apart from the plan, labelled, because the Eligibility
  Agent's check of it covers the sections it cites and the outcome it states and not what it says
  about the client. A test reads this module and fails if it ever touches `explanation`.
- **A blocked statement is shown as blocked, with its reason.** A statement with no source, an
  unknown PATS section, a stale source or a conflict is kept in `state.plan` with status
  `blocked` and the reason in words the user can read. It is never silently dropped and never
  shown as verified.
- **A blocked core statement sends the plan back.** The statements the whole plan rests on (the
  service, the outcome, the travel mode, each option) are *core*. If one is blocked the plan cannot
  be shown with a hole in it, so the run goes to the Coordinator to re-plan (the design diagram: "fails
  check -> re-plan"), and the run ends with the reason if no re-plan is possible.
- **Uncertainty goes to the user.** Stale or conflicting information becomes a question at the plan
  gate ("this was last checked on ... please confirm it is still current"), together with the
  questions the other tools raised.
- **Failures go to staff with a fixed reason**, never the exception text. A bug (a state
  with nothing to check) raises instead.

The trace counts statements and checks; it never repeats a statement, because a statement can hold
the client's town or specialty.
"""

from typing import Any, Literal

from communitybridge.config import ConfigError
from communitybridge.graph.node_contract import NodeFn, traced_node
from communitybridge.graph.state import GraphState
from communitybridge.graph.toolbox import ToolBox
from communitybridge.models import TraceEvent
from communitybridge.models.replan import window_for
from communitybridge.models.verification import PlanCheck
from communitybridge.tools.service_query import ServiceQueryError

# The tool's name in the permission map (`AGENT_TOOLS` in graph/build.py) and in the trace.
PLAN_VERIFIER = "plan_verifier"


class VerifierAgentError(Exception):
    """The graph asked this agent to do something it should never be asked to do (a bug)."""


def _event(task: str, result: str, status: Any) -> TraceEvent:
    return TraceEvent(
        task=task, agent="verifier", tool_or_source=PLAN_VERIFIER, result=result, status=status
    )


def _escalate(reason: str) -> dict[str, Any]:
    """End the run for staff. Two events: what failed, and the hand-over."""
    return {
        "status": "escalated",
        "trace": [
            _event("verify plan", reason, "failed"),
            TraceEvent(
                task="pass to staff",
                agent="verifier",
                tool_or_source="escalation to staff",
                result="the plan could not be checked; a staff member can take the request",
                status="escalated",
            ),
        ],
    }


def _summary(check: PlanCheck) -> str:
    report = check.report
    total, blocked = len(report.results), len(report.blocked)
    summary = f"{total} statements checked: {total - blocked} verified, {blocked} blocked"
    if blocked:
        kinds = [f.check for result in report.blocked for f in result.findings]
        summary += " (" + ", ".join(f"{kinds.count(k)} {k}" for k in dict.fromkeys(kinds)) + ")"
    if report.replan_needed:
        summary += (
            "; a statement the plan rests on is blocked, so the plan goes back to be re-planned"
        )
    if check.questions:
        summary += f"; {len(check.questions)} question(s) for the user"
    return summary


def make_verify_plan(tools: ToolBox) -> NodeFn:
    """Build the verify_plan node around the tool box for this agent."""

    @traced_node("verifier")
    def verify_plan(state: GraphState) -> dict[str, Any]:
        goal = state.goal
        if goal is None:
            raise VerifierAgentError("the state has no goal to verify a plan for")
        if not goal.confirmed_by_user:
            raise VerifierAgentError("the goal has not been confirmed by the user (Gate 1)")
        if state.service is None or state.eligibility is None:
            raise VerifierAgentError("the state has no service and eligibility result to verify")
        try:
            check: PlanCheck = tools.call(
                PLAN_VERIFIER,
                state.service,
                state.eligibility,
                state.reasons,
                state.logistics,
                state.options,
                state.fallback,
                window_for(goal, state.fallback),
            )
        except ConfigError:
            return _escalate("the verification settings could not be used")
        except ServiceQueryError:
            return _escalate("the service database could not be read")
        report = check.report
        return {
            "plan": report.items,
            "questions": check.questions,
            "replan_needed": report.replan_needed,
            "trace": [
                _event("verify plan", _summary(check), "ok" if report.all_verified else "blocked")
            ],
        }

    return verify_plan


def route_after_verification(state: GraphState) -> Literal["approve", "replan", "finish"]:
    """A plan that can be shown goes to Gate 2. One with a blocked core statement goes back to be
    re-planned. An escalated run finishes."""
    if state.status == "escalated":
        return "finish"
    if not state.plan:
        raise VerifierAgentError("verification ended with no plan and was not escalated")
    return "replan" if state.replan_needed else "approve"
