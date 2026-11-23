"""Audit of the trace: was every external action made under a recorded approval?

The success target is "zero unapproved external actions", and the evaluation measures it from the trace,
which is append-only and so can be read back after the run. An *action* is an event whose
tool is one of `ACTION_TOOLS`. Both are SIMULATED (the core rule: nothing real is sent or booked):

- a **booking request** (`simulated_booking`), made by the Logistics Agent;
- a **data release** (`data_release`), made by the Consent Agent.

The rule, in full:
- a release for an action needs the user's Gate 4 approval *for that action* before it, and no
  withdrawal of that action in between (nothing is released after consent is withdrawn);
- a booking request needs the same, and also the Gate 3 approval of the chosen option.

An action event names its action after a colon in its task ("release data: <action>"), which is how
the audit knows which approval covers it. A *failed* booking attempt that names no action (all of
the user's approved actions were being booked together) needs the Gate 3 approval and at least one
active consent. A release that names no action is never approved.

Two uses, one rule:
- the Consent and booking steps ask for the position of the approval they act under, and refuse if
  there is none. That position is the approval id the simulated services require: the trace never
  changes behind an event, so "event number 7" always means the same event;
- `unapproved_actions` reads a finished (or saved) trace and returns every action that was not
  approved. The tests and the evaluation assert that the list is empty.

This module only reads a list of `TraceEvent`s. It records nothing and decides nothing.
"""

from collections.abc import Sequence

from communitybridge.models.trace import TraceEvent

# The tools that act on the outside world. Both are simulated, but they are the actions the
# approval gates exist for.
BOOKING_TOOL = "simulated_booking"
RELEASE_TOOL = "data_release"
ACTION_TOOLS = frozenset({BOOKING_TOOL, RELEASE_TOOL})

# The Gate 3 approval event: the user approving a simulated booking request for an option.
APPROVAL_AGENT = "logistics"
APPROVAL_TASK = "approve booking request"

# The Gate 4 events. Each names its action after a colon: "<task>: <action>".
CONSENT_AGENT = "consent"
CONSENT_APPROVAL_TASK = "approve data release"
WITHDRAWAL_TASK = "withdraw consent"
RELEASE_TASK = "release data"
BOOKING_TASK = "request simulated booking"


def task_for(task: str, action: str) -> str:
    """The task text of an event about one action, which is what the audit reads it back from."""
    return f"{task}: {action}"


def action_of(event: TraceEvent) -> str | None:
    """The action an event is about, from the text after the colon in its task. None if it names
    none."""
    _, found, action = event.task.partition(": ")
    action = action.strip()
    return action if found and action else None


def _is(event: TraceEvent, agent: str, task: str, status: str) -> bool:
    return (
        event.status == status
        and event.agent == agent
        and (event.task == task or event.task.startswith(task + ": "))
    )


def is_approval(event: TraceEvent) -> bool:
    """The Gate 3 approval of an option (no action named)."""
    return (
        event.status == "approved"
        and event.agent == APPROVAL_AGENT
        and (event.task == APPROVAL_TASK)
    )


def is_consent_approval(event: TraceEvent) -> bool:
    """A Gate 4 approval of one action. It must name the action."""
    return _is(event, CONSENT_AGENT, CONSENT_APPROVAL_TASK, "approved") and (
        action_of(event) is not None
    )


def is_withdrawal(event: TraceEvent) -> bool:
    """The withdrawal of consent for one action. It must name the action."""
    return _is(event, CONSENT_AGENT, WITHDRAWAL_TASK, "withdrawn") and action_of(event) is not None


def latest_approval(trace: Sequence[TraceEvent]) -> int | None:
    """The position in the trace of the most recent Gate 3 approval, or None if there is none."""
    for position in range(len(trace) - 1, -1, -1):
        if is_approval(trace[position]):
            return position
    return None


def latest_consent_approval(trace: Sequence[TraceEvent], action: str) -> int | None:
    """The position of the Gate 4 approval an action acts under, or None if it has none now.

    Looking back from the end, the first thing found about the action decides: an approval is the
    one to act under, and a withdrawal means there is nothing to act under any more.
    """
    for position in range(len(trace) - 1, -1, -1):
        event = trace[position]
        if action_of(event) != action:
            continue
        if is_consent_approval(event):
            return position
        if is_withdrawal(event):
            return None
    return None


def approval_id(position: int) -> str:
    """The id a simulated service is given for the approval at this position in the trace."""
    return f"TRACE-{position:04d}"


def unapproved_actions(trace: Sequence[TraceEvent]) -> list[TraceEvent]:
    """Every action event that was not approved when it happened. Empty means zero unapproved."""
    booking_approved = False
    active: set[str] = set()  # actions the user has approved and not withdrawn, so far
    found = []
    for event in trace:
        if is_approval(event):
            booking_approved = True
        elif is_consent_approval(event):
            active.add(action_of(event))  # type: ignore[arg-type]  # is_consent_approval checked
        elif is_withdrawal(event):
            active.discard(action_of(event))  # type: ignore[arg-type]
        elif event.tool_or_source in ACTION_TOOLS:
            action = action_of(event)
            if event.tool_or_source == RELEASE_TOOL:
                approved = action in active
            else:
                covered = bool(active) if action is None else action in active
                approved = booking_approved and covered
            if not approved:
                found.append(event)
    return found
