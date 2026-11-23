"""The Consent Agent: shows the minimum data each action needs, waits for approval, and lets consent
be withdrawn.

The project design: the Consent Agent "shows the minimum data each action needs", its data is
`data/policy/data_fields.yaml` only, its check is "rule-based", and it has a human gate: "Yes,
explicit, per action". The project plan says the system will "show the minimum personal data each
action needs, wait for approval and allow consent to be withdrawn". The design diagram draws this
agent without a model, so, like the Verifier, it is exactly as originally designed. This module has
three parts, in the order the run reaches them:

1. Gate 4 (`announce_approve_release`, `approve_release`) shows, for each action the chosen option
   needs (the appointment, the transport, the stay), the policy's fields with their values, who
   receives them and why, and the user approves or declines **each action separately**.
2. `release_data` hands the minimum fields to each SIMULATED recipient whose request was made
   for an action the user approved, and to no other. It runs after the booking step on purpose:
   the booking is all-or-nothing and is the step that can fail (a seat may have gone since the
   options were shown), and no details should leave for a request that could not be made.
3. `withdraw_consent` is the way back: it undoes one action's release and cancels its
   SIMULATED hold. It is not on the main path; the interface applies it to a finished run.

Design points worth being able to explain:

- **Explicit and per action.** The answer must decide every action offered, one by one; there is
  no "approve all" and no default. Declining an action, or never answering, releases nothing for it.
  An action whose details the system does not hold cannot be approved: the user is shown what is
  missing and the gate asks again, so nobody approves a blank.
- **What was approved is what is sent.** The consent screen and the release are built from the same
  function (`tools/consent_flow.py`), so the values the user saw are the values released.
- **Minimum data, in the policy.** An action's fields come from the policy and nowhere else; the
  policy refuses to load if an action uses a sensitive or never-share field for a provider.
- **The agent never reads a personal value.** It passes the goal, service, option and contact to its
  tools and gets screens and records back. The trace names actions, recipients and field *names*,
  never values, because a value can be a name or a date of birth.
- **No release without an approval in the trace, enforced in layers.** The release node raises (it
  does not escalate) unless the trace holds the Gate 4 approval for that action, and gives the
  SIMULATED sink the position of that approval as the approval id; the consent tool refuses unless
  the consent is active; and the sink refuses an empty id. `trace/audit.py` reads any finished
  trace and lists releases and bookings with no approval before them, or after a withdrawal.
- **All or nothing.** Before any release is made the node checks that every approved action has
  all its details, so a gap cannot leave one recipient with data and another without.
- **Failures go to staff with a fixed reason**, never the exception text. A bug raises instead.

Out of scope here: sharing the plan with an ACCHO officer (`share_plan_with_authorised_officer`
in the policy) has no path in the state graph yet, so nothing releases sensitive fields.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from langgraph.types import interrupt
from pydantic import Field, ValidationError

from communitybridge.graph.node_contract import NodeFn, describe_problems, traced_node
from communitybridge.graph.state import GraphState
from communitybridge.graph.toolbox import ToolBox
from communitybridge.models import StrictModel, TraceEvent
from communitybridge.models.consent import KIND_FOR_ACTION
from communitybridge.models.consent_release import ConsentSubject, ReleaseRecord, WithdrawalResult
from communitybridge.tools.consent import ConsentError
from communitybridge.tools.policy_loader import PolicyDataError
from communitybridge.trace.audit import (
    CONSENT_APPROVAL_TASK,
    RELEASE_TASK,
    RELEASE_TOOL,
    WITHDRAWAL_TASK,
    approval_id,
    latest_consent_approval,
    task_for,
)

# The tools' names in the permission map (`AGENT_TOOLS` in graph/build.py) and in the trace.
CONSENT_ACTIONS = "consent_actions"
CONSENT_PROMPT = "consent_prompt"
CONSENT_REQUEST = "consent_request"
CONSENT_DECISION = "consent_decision"
DATA_RELEASE = RELEASE_TOOL
CONSENT_WITHDRAWAL = "consent_withdrawal"

GATE_FOUR = "approve_release"


class ConsentAgentError(Exception):
    """The graph asked this agent to do something it should never be asked to do (a bug)."""


class GateFourAnswerError(ConsentAgentError):
    """The answer sent back for Gate 4 was not one the gate understands."""


class ConsentApprovalMissingError(ConsentAgentError):
    """Data was about to be released with no approval in the trace. It never is."""


class GateFourAnswer(StrictModel):
    """What the interface sends back when the user answers Gate 4: one decision per action.

    Every action offered must be decided; "approve" releases the minimum data for that action only,
    "decline" releases nothing for it.
    """

    decisions: dict[str, Literal["approve", "decline"]] = Field(min_length=1)


def parse_gate_four_answer(
    raw: object, offered: Sequence[str], missing: Mapping[str, Sequence[str]]
) -> GateFourAnswer:
    """Check the answer from the interface. A bad one leaves the run waiting at the gate.

    The message names fields and actions from the policy, never anything the user typed.
    """
    try:
        answer = GateFourAnswer.model_validate(raw)
    except ValidationError as exc:
        raise GateFourAnswerError(
            f"the answer to Gate 4 (approve release) was not understood: {describe_problems(exc)}"
        ) from exc
    if set(answer.decisions) != set(offered):
        raise GateFourAnswerError(
            "the answer to Gate 4 (approve release) was not understood: decisions: must decide "
            f"exactly the actions offered ({', '.join(offered)}), one by one"
        )
    for action in offered:
        if answer.decisions[action] == "approve" and missing.get(action):
            raise GateFourAnswerError(
                f"the answer to Gate 4 (approve release) was not understood: decisions: "
                f"{action} cannot be approved while these details are missing: "
                f"{', '.join(missing[action])}"
            )
    return answer


def _event(task: str, source: str, result: str, status: Any) -> TraceEvent:
    return TraceEvent(
        task=task, agent="consent", tool_or_source=source, result=result, status=status
    )


def _escalate(task: str, source: str, reason: str) -> dict[str, Any]:
    """End the run for staff. Two events: what failed, and the hand-over."""
    return {
        "status": "escalated",
        "trace": [
            _event(task, source, reason, "failed"),
            _event(
                "pass to staff",
                "escalation to staff",
                "no data was released; a staff member can take the request",
                "escalated",
            ),
        ],
    }


def _subject(state: GraphState) -> ConsentSubject:
    """The journey the user chose, which is what the Consent tools fill an action's fields from."""
    if state.goal is None or not state.goal.confirmed_by_user:
        raise ConsentAgentError("there is no confirmed goal to ask consent for")
    if state.service is None or state.chosen_option is None:
        raise ConsentAgentError("there is no chosen option to ask consent for")
    return ConsentSubject(
        goal=state.goal,
        service=state.service,
        option=state.options[state.chosen_option],
        contact=state.contact,
    )


# --- 1. Gate 4: the user approves, action by action, what is released -----------------------------


@traced_node("consent")
def announce_approve_release(state: GraphState) -> dict[str, Any]:
    """The node before the gate: records that the run is waiting for the user."""
    return {
        "status": "awaiting_approval",
        "trace": [
            _event(
                CONSENT_APPROVAL_TASK,
                "gate 4",
                "waiting for the user to approve, action by action, the minimum data each needs",
                "awaiting_approval",
            )
        ],
    }


def make_approve_release(tools: ToolBox) -> NodeFn:
    """Build the Gate 4 node around the tool box for this agent."""

    @traced_node("consent")
    def approve_release(state: GraphState) -> dict[str, Any]:
        subject = _subject(state)
        task = CONSENT_APPROVAL_TASK
        try:
            actions: list[str] = tools.call(CONSENT_ACTIONS, subject.option)
            prompts = [tools.call(CONSENT_PROMPT, action, subject) for action in actions]
        except (ConsentError, PolicyDataError):
            return _escalate(task, CONSENT_PROMPT, "the consent policy could not be used")
        payload: dict[str, Any] = {
            "gate": GATE_FOUR,
            "actions": [prompt.model_dump(mode="json") for prompt in prompts],
        }
        missing = {prompt.action: prompt.missing for prompt in prompts}
        while True:
            raw = interrupt(payload)
            try:
                answer = parse_gate_four_answer(raw, actions, missing)
                break
            except GateFourAnswerError as exc:
                # As at Gate 1: LangGraph replays a stored answer, so raising here would trap the
                # run. Ask again, saying which field was wrong (names only, never the value).
                payload = {**payload, "error": str(exc)}

        decided, events = [], []
        for action in actions:
            approved = answer.decisions[action] == "approve"
            request = tools.call(CONSENT_REQUEST, action)
            decided.append(tools.call(CONSENT_DECISION, request, approved))
            events.append(
                _event(
                    task_for(CONSENT_APPROVAL_TASK, action),
                    "user",
                    "the user approved releasing the minimum data for this action"
                    if approved
                    else "the user declined; no data will be released for this action",
                    "approved" if approved else "blocked",
                )
            )
        return {
            "consents": decided,
            "status": "in_progress" if any(c.is_active for c in decided) else "declined",
            "trace": events,
        }

    return approve_release


def route_after_release_gate(state: GraphState) -> Literal["book", "finish"]:
    """Every action declined ends the run with nothing requested or released. Otherwise the
    SIMULATED requests are made for the approved actions, and the data is released after them."""
    if state.status in ("declined", "escalated"):
        return "finish"
    if not any(consent.is_active for consent in state.consents):
        raise ConsentAgentError("Gate 4 ended with no approved action and was not declined")
    return "book"


# --- 2. the SIMULATED release, only for what the user approved ------------------------------------


def make_release_data(tools: ToolBox) -> NodeFn:
    """Build the release node around the tool box for this agent."""

    @traced_node("consent")
    def release_data(state: GraphState) -> dict[str, Any]:
        subject = _subject(state)
        done = {record.release.action for record in state.releases}
        # Data goes only to a recipient whose request was made: the booking comes first.
        requested = {b.kind for b in state.bookings if b.status == "requested"}
        pending = [
            c
            for c in state.consents
            if c.is_active
            and c.action not in done
            and (c.action not in KIND_FOR_ACTION or KIND_FOR_ACTION[c.action] in requested)
        ]
        if not pending:
            raise ConsentAgentError("there is no approved action left to release data for")
        # The check that matters most in this module: nothing is released unless the trace shows
        # the user approved that action. It raises, so a missing approval is a bug that stops the
        # run and is never passed on as if it were a problem with the client's request.
        positions = {}
        for consent in pending:
            position = latest_consent_approval(state.trace, consent.action)
            if position is None:
                raise ConsentApprovalMissingError(
                    "no data is released without an approval in the trace"
                )
            positions[consent.action] = position
        task = "release data"
        try:
            # All or nothing: make sure every approved action has all its details before any
            # recipient is given anything.
            gaps = [tools.call(CONSENT_PROMPT, c.action, subject).missing for c in pending]
        except (ConsentError, PolicyDataError):
            return _escalate(task, CONSENT_PROMPT, "the consent policy could not be used")
        if any(gaps):
            return _escalate(
                task, CONSENT_PROMPT, "a detail needed for an approved action is missing"
            )
        records: list[ReleaseRecord] = []
        for consent in pending:
            records.append(
                tools.call(DATA_RELEASE, consent, subject, approval_id(positions[consent.action]))
            )
        return {
            "releases": [*state.releases, *records],
            "trace": [
                _event(
                    task_for(RELEASE_TASK, record.release.action),
                    DATA_RELEASE,
                    f"SIMULATED release {record.reference} to {record.release.recipient}: "
                    f"{', '.join(record.release.data)} under approval {record.approval_id}",
                    "ok",
                )
                for record in records
            ],
        }

    return release_data


# --- 3. withdrawing consent for one action ----------------------------------------------


def make_withdraw_consent(tools: ToolBox, action: str) -> NodeFn:
    """Build the withdrawal node for one action.

    It is applied to a finished run by the interface (`withdraw_consent` in graph/build.py), not
    reached by an edge: withdrawing is the user's choice, at any time after approving.
    """

    @traced_node("consent")
    def withdraw_consent(state: GraphState) -> dict[str, Any]:
        request = next((c for c in state.consents if c.action == action), None)
        if request is None:
            raise ConsentAgentError(f"there is no consent for {action} to withdraw")
        # The tool refuses (ConsentStateError) unless the consent is approved and not withdrawn,
        # and leaves the SIMULATED log untouched when it does.
        result: WithdrawalResult = tools.call(CONSENT_WITHDRAWAL, request)
        cancelled = {booking.reference: booking for booking in result.cancelled}
        return {
            "consents": [result.request if c.action == action else c for c in state.consents],
            "releases": [r for r in state.releases if r.release.action != action],
            "bookings": [cancelled.get(b.reference, b) for b in state.bookings],
            "withdrawals": [*state.withdrawals, result.notice],
            "trace": [
                _event(
                    task_for(WITHDRAWAL_TASK, action),
                    CONSENT_WITHDRAWAL,
                    f"consent withdrawn for {action}: the released details were removed and "
                    f"{len(result.cancelled)} SIMULATED request(s) cancelled",
                    "withdrawn",
                )
            ],
        }

    return withdraw_consent
