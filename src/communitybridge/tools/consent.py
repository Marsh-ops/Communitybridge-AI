"""Consent tool: the minimum data each action needs, per-action approval, and withdrawal.

The Consent Agent's only data source is `data/policy/data_fields.yaml`, and its
check is "rule-based", with a human gate on every action. The project plan says the system shows the
minimum personal data each action needs, waits for approval and lets consent be withdrawn. This module is where that happens, and it is the only way data leaves the system, so the
"zero unapproved external actions" target comes down to the rules below:

1. **Minimum data.** An action's fields come from the policy and nowhere else. A request that
   asks for more (or fewer) fields than the policy names for that action is refused, and fields
   the system holds but the policy doesn't list (a Medicare number, clinical notes) are never
   released, however they got into `available`.
2. **Approval is per action and recorded.** `request_consent` makes a pending request,
   `record_decision` stamps the decision once, and `release_data` refuses unless the request is
   active (approved and not withdrawn). Declining, or never answering, releases nothing.
3. **What the user saw is what is sent.** The consent screen (`describe_action`) and the release
   (`release_data`) format values with the same function, so the data cannot differ from what was
   approved.
4. **Withdrawal.** `withdraw_consent` returns a new record (the approval stays on record) and a
   notice that says, from the policy, what withdrawing undoes for that action.

Nothing is really sent: every recipient is SIMULATED or a prospective partner, so each result carries `simulated: true`. Errors name fields, never values, so a message in
the agent trace can't leak personal data. Time is passed in (`at`) instead of read from the clock,
which keeps the tool pure and its tests exact.
"""

from collections.abc import Mapping
from datetime import date, datetime

from communitybridge.models.consent import ConsentRequest
from communitybridge.models.consent_policy import ConsentAction, ConsentPolicy
from communitybridge.models.consent_release import (
    ConsentPrompt,
    DataRelease,
    PromptField,
    WithdrawalNotice,
)


class ConsentError(Exception):
    """Base of every Consent tool failure, so the agent can catch them together and trace them."""


class UnknownActionError(ConsentError):
    """The action isn't defined in the policy, so there is no minimum data to show or release."""


class ConsentStateError(ConsentError):
    """The change isn't allowed from the request's current state (e.g. deciding twice)."""


class ConsentRequiredError(ConsentError):
    """Data was asked for without an active consent: still pending, declined or withdrawn."""


class ConsentMismatchError(ConsentError):
    """The request doesn't ask for exactly the fields the policy names for its action."""


class MissingDataError(ConsentError):
    """An approved action needs a field the system doesn't have; nothing is released."""


def _spec(action_id: str, policy: ConsentPolicy) -> ConsentAction:
    action = policy.actions.get(action_id)
    if action is None:
        raise UnknownActionError(f"'{action_id}' is not an action in the consent policy")
    return action


def _as_text(value: object) -> str | None:
    """Show a value the way the user reads it. None means the system doesn't have it."""
    if value is None:
        return None
    if isinstance(value, bool):  # before any other check: a bool is also an int
        return "yes" if value else "no"
    if isinstance(value, date):  # covers datetime too
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _values(action: ConsentAction, available: Mapping[str, object]) -> dict[str, str | None]:
    """The action's fields only, in policy order. Anything else in `available` is left behind."""
    return {name: _as_text(available.get(name)) for name in action.required_fields}


def describe_action(
    action_id: str, policy: ConsentPolicy, available: Mapping[str, object] | None = None
) -> ConsentPrompt:
    """Build the consent screen: the minimum fields (with values), who gets them and why."""
    action = _spec(action_id, policy)
    values = _values(action, available or {})
    return ConsentPrompt(
        action=action_id,
        description=action.description,
        recipient=action.recipient,
        recipient_type=action.recipient_type,
        purpose=action.purpose,
        fields=[
            PromptField(
                name=name,
                description=policy.fields[name].description,
                sensitivity=policy.fields[name].sensitivity,
                value=value,
            )
            for name, value in values.items()
        ],
        missing=[name for name, value in values.items() if value is None],
        on_withdrawal=action.on_withdrawal,
    )


def request_consent(action_id: str, policy: ConsentPolicy) -> ConsentRequest:
    """Start a pending request for one action, carrying the policy's fields and purpose."""
    action = _spec(action_id, policy)
    return ConsentRequest(
        action=action_id, fields_required=list(action.required_fields), purpose=action.purpose
    )


def record_decision(request: ConsentRequest, approved: bool, at: datetime) -> ConsentRequest:
    """Record the user's answer once. A decision is final, so a changed mind is a new request."""
    if request.status != "pending":
        raise ConsentStateError(
            f"consent for {request.action} is already {request.status}; start a new request"
        )
    return ConsentRequest.model_validate(
        {**request.model_dump(), "approved": approved, "decided_at": at}
    )


def release_data(
    request: ConsentRequest, available: Mapping[str, object], policy: ConsentPolicy
) -> DataRelease:
    """Release the policy's fields for this action, only while its consent is active."""
    action = _spec(request.action, policy)
    if not request.is_active:
        raise ConsentRequiredError(
            f"no data released for {request.action}: consent is {request.status}"
        )
    # Approval is per action. If the request's fields aren't exactly the policy's, the user
    # approved something else than what would be sent.
    if sorted(request.fields_required) != sorted(action.required_fields):
        raise ConsentMismatchError(
            f"consent request for {request.action} lists {request.fields_required}, "
            f"but the policy allows {action.required_fields}"
        )
    values = _values(action, available)
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise MissingDataError(f"{request.action} needs {', '.join(missing)}; nothing released")
    return DataRelease(
        action=request.action,
        recipient=action.recipient,
        recipient_type=action.recipient_type,
        data={name: value for name, value in values.items() if value is not None},
        # An active request always has decided_at: ConsentRequest rejects an approval without it.
        consent_decided_at=request.decided_at,  # type: ignore[arg-type]
    )


def withdraw_consent(
    request: ConsentRequest, at: datetime, policy: ConsentPolicy
) -> tuple[ConsentRequest, WithdrawalNotice]:
    """Withdraw an approved consent. Returns the new record and what withdrawing undoes."""
    action = _spec(request.action, policy)
    if request.status == "withdrawn":
        raise ConsentStateError(f"consent for {request.action} is already withdrawn")
    if request.status != "approved":
        raise ConsentStateError(
            f"only an approved consent can be withdrawn; consent for {request.action} is "
            f"{request.status}"
        )
    try:
        withdrawn = request.withdraw(at)
    except ValueError as exc:  # the model rejects a withdrawal dated before the approval
        raise ConsentStateError(f"consent for {request.action} can't be withdrawn: {exc}") from exc
    notice = WithdrawalNotice(
        action=request.action,
        recipient=action.recipient,
        withdrawn_at=at,
        message=action.on_withdrawal,
    )
    return withdrawn, notice
